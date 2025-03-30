import os
import re
import time
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec

# 
# ======================
# 配置参数（根据需求调整）
# ======================
BASE_DIR = r"E:\Download\adrive\数字创业活力识别\2.种子库"
SEED_XLSX_PATH = os.path.join(BASE_DIR, "种子词筛选.xlsx")
CORPUS_DIR = os.path.join(BASE_DIR, "1.Word2Vce语料库_")
OUTPUT_DIR = os.path.join(BASE_DIR, "分类结果")
MODEL_PATH = os.path.join(BASE_DIR, "word2vec.model")
STATUS_DB = os.path.join(OUTPUT_DIR, "processing.db")
VALIDATION_DIR = os.path.join(BASE_DIR, "验证报告")
INTERFERENCE_THRESHOLD = 0.28  # 更严格的干扰阈值
BATCH_SIZE = 100  # 减小批次规模
MAX_MEMORY_PERCENT = 90
MODEL_MMAP = True
MODEL_CACHE_SIZE = 2000  # 缓存容量

class TqdmProgress(CallbackAny2Vec):
    """训练进度监视器"""
    def __init__(self, total_epochs, desc="训练进度"):
        self.pbar = None
        self.total_epochs = total_epochs
        self.desc = desc
        
    def on_train_begin(self, model):
        self.pbar = tqdm(total=self.total_epochs, desc=self.desc)
        
    def on_epoch_end(self, model):
        self.pbar.update(1)
        current_loss = model.get_latest_training_loss()
        self.pbar.set_postfix({"loss": f"{current_loss:,.1f}"})
        
    def on_train_end(self, model):
        self.pbar.close()

# ======================
# 初始化数据库
# ======================
def init_database():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(STATUS_DB, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS processed (
        category TEXT,
        keyword TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (category, keyword)
    ) WITHOUT ROWID""")
    conn.commit()
    conn.close()

# ======================
# 语料预处理模块（无分词）
# ======================
def preprocess_corpus():
    corpus_file = os.path.join(CORPUS_DIR, "combined_corpus.txt")
    
    if not os.path.exists(corpus_file):
        txt_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
        
        with open(corpus_file, "w", encoding="utf-8") as f_out:
            with tqdm(total=len(txt_files), desc="合并语料") as pbar:
                for filename in txt_files:
                    filepath = os.path.join(CORPUS_DIR, filename)
                    with open(filepath, encoding="utf-8") as f_in:
                        for line in f_in:
                            # 统一符号处理流程
                            processed_line = line.strip()
                            
                            # 第一步：替换所有非字母数字汉字字符为空格
                            processed_line = re.sub(r'[^\w\u4e00-\u9fff]', ' ', processed_line)
                            
                            # 第二步：合并连续空格
                            processed_line = re.sub(r'\s+', ' ', processed_line)
                            
                            # 第三步：去除首尾空格
                            processed_line = processed_line.strip()
                            
                            if processed_line:
                                # 直接写入处理后的整行
                                f_out.write(processed_line + '\n')
                    pbar.update(1)
    return corpus_file

# ======================
# 分类体系管理器
# ======================
class HierarchyManager:
    def __init__(self, model):
        self.model = model
        self.hierarchy = self._load_hierarchy()
    
    def _load_hierarchy(self):
        df = pd.read_excel(SEED_XLSX_PATH)
        return df.groupby('中类')['小类'].apply(lambda x: ' '.join(x).split()).to_dict()
    
    @lru_cache(maxsize=MODEL_CACHE_SIZE)
    def get_interference(self, phrase, current_cat):
        if phrase not in self.model.wv:
            return 0.0
        
        target_vec = self.model.wv[phrase]
        other_vectors = []
        for cat, words in self.hierarchy.items():
            if cat != current_cat:
                other_vectors.extend([self.model.wv[w] for w in words if w in self.model.wv])
        
        if not other_vectors:
            return 0.0
        
        similarities = np.dot(other_vectors, target_vec) / (
            np.linalg.norm(other_vectors, axis=1) * np.linalg.norm(target_vec)
        )
        return np.mean(similarities)

# ======================
# 资源管理模块
# ======================
def get_safe_workers():
    total_mem = psutil.virtual_memory().total / (1024**3)
    free_mem = psutil.virtual_memory().available / (1024**3)
    safe_mem = free_mem - 3
    
    if safe_mem < 1:
        return 1
    
    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH) / (1024**2)
        est_mem = model_size * 1.2 / 1024
        max_workers = int(safe_mem // est_mem)
        return max(1, min(max_workers, os.cpu_count()))
    return os.cpu_count()

# ======================
# 核心处理流程
# ======================
def memory_aware_process(args):
    batch, ctx = args
    
    try:
        model = Word2Vec.load(ctx['model_path'], mmap=ctx['model_mmap'] and 'r' or None)
        hm = HierarchyManager(model)
        conn = sqlite3.connect(ctx['status_db'], timeout=30)
        
        results = []
        validation_data = []
        pattern = re.compile(r'数字|智能|云|物联')  # 关键特征匹配模式
        
        for category, keyword in batch:
            if conn.execute("SELECT 1 FROM processed WHERE category=? AND keyword=?", 
                          (category, keyword)).fetchone():
                continue
            
            try:
                similar_phrases = model.wv.most_similar(
                    positive=[keyword],
                    topn=30,
                    restrict_vocab=50000
                )
                
                batch_results = []
                total_sim = 0.0
                valid_count = 0
                
                for phrase, similarity in similar_phrases:
                    # 过滤条件
                    if len(phrase) < 4:  # 过滤短词
                        continue
                    
                    interference = hm.get_interference(phrase, category)
                    is_valid = (similarity > 0.65 
                               and interference < ctx['threshold']
                               and bool(pattern.search(phrase)))
                    
                    total_sim += similarity
                    if is_valid:
                        valid_count += 1
                    
                    batch_results.append((
                        category, keyword, phrase, 
                        round(similarity,4), 
                        round(interference,4), 
                        is_valid
                    ))
                
                if batch_results:
                    results.extend(batch_results)
                    validation_data.append({
                        "category": category,
                        "keyword": keyword,
                        "valid_ratio": valid_count / len(batch_results),
                        "avg_similarity": total_sim / len(batch_results),
                        "timestamp": pd.Timestamp.now().isoformat()
                    })
                
                conn.execute(
                    "INSERT OR IGNORE INTO processed (category, keyword) VALUES (?, ?)", 
                    (category, keyword)
                )
                conn.commit()
                
            except KeyError:
                continue
            except Exception as e:
                print(f"处理异常 {category}/{keyword}: {str(e)}")
        
        conn.close()
        return results, validation_data
    except Exception as e:
        print(f"进程异常: {str(e)}")
        return [], []

# ======================
# 模型训练模块
# ======================
def train_word2vec(corpus_file):
    if not os.path.exists(MODEL_PATH):
        # 动态调整训练轮次
        corpus_size = sum(1 for _ in open(corpus_file, encoding='utf-8'))
        base_epochs = max(25, min(40, 2000000 // corpus_size))  # 增加基准训练轮次
        
        progress = TqdmProgress(total_epochs=base_epochs)
        
        model = Word2Vec(
            sentences=LineSentence(corpus_file),
            vector_size=400,          # 增大向量维度          #400
            window=10,                # 扩大上下文窗口        #15
            min_count=5,              # 提高低频词过滤        #5
            workers=os.cpu_count()-1,
            epochs=base_epochs,
            sg=1,                     # 保持skip-gram架构
            hs=0,                     # 禁用层次softmax
            negative=15,              # 增加负采样数量        #15
            ns_exponent=0.75,         # 负采样分布指数        #0.75
            sample=1e-4,              # 高频词降采样          #1e-5
            alpha=0.025,               # 初始学习率            #0.03
            min_alpha=0.0001,         # 最小学习率            #0.0007
            compute_loss=True,
            callbacks=[progress],
            max_vocab_size=800000     # 限制词表规模
        )
        
        # 添加增量训练（可选）
        if corpus_size < 1000000:     #100000
            model.train(
                LineSentence(corpus_file),
                total_examples=model.corpus_count,
                epochs=10,  # 额外训练轮次                     #5                            
                compute_loss=True
            )
        
        model.save(MODEL_PATH)
    return Word2Vec.load(MODEL_PATH, mmap=MODEL_MMAP and 'r' or None)

# ======================
# 验证报告生成
# ======================
def generate_validation_report():
    report_data = []
    pattern = re.compile(r'数字|智能|云|物联')
    
    for cat in pd.read_excel(SEED_XLSX_PATH)['中类'].unique():
        safe_cat = re.sub(r'[\\/*?:"<>|]', '_', cat)
        csv_path = os.path.join(OUTPUT_DIR, f"{safe_cat}.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
            
        valid_ratio = df['is_valid'].mean()
        feature_match = df['similar_word'].str.contains(pattern).mean()
        
        report_data.append({
            '中类': cat,
            '有效匹配率': round(valid_ratio, 4),
            '特征覆盖率': round(feature_match, 4),
            '示例匹配': ' | '.join(df[df['is_valid']].head(5)['similar_word']),
            '平均相似度': round(df['similarity'].mean(), 4),
            '平均干扰度': round(df['interference'].mean(), 4)
        })
    
    pd.DataFrame(report_data).to_excel(
        os.path.join(VALIDATION_DIR, "质量验证报告.xlsx"), index=False)

# ======================
# 主程序
# ======================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    init_database()

    corpus_file = preprocess_corpus()
    model = train_word2vec(corpus_file)
    hm = HierarchyManager(model)
    
    tasks = [(cat, kw) for cat, kws in hm.hierarchy.items() for kw in kws]
    batches = [tasks[i:i+BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
    
    ctx = {
        'model_path': MODEL_PATH,
        'status_db': STATUS_DB,
        'max_mem': MAX_MEMORY_PERCENT,
        'model_mmap': MODEL_MMAP,
        'threshold': INTERFERENCE_THRESHOLD
    }
    
    with ProcessPoolExecutor(max_workers=get_safe_workers()) as executor:
        futures = [executor.submit(memory_aware_process, (batch, ctx)) for batch in batches]
        
        with tqdm(total=len(futures), desc="处理进度") as pbar:
            for future in futures:
                while psutil.virtual_memory().percent > MAX_MEMORY_PERCENT:
                    time.sleep(5)
                    print(f"内存使用率 {psutil.virtual_memory().percent}%，等待释放...")
                
                results, valid_data = future.result()
                
                if results:
                    df = pd.DataFrame(results, columns=[
                        'category', 'keyword', 'similar_word', 
                        'similarity', 'interference', 'is_valid'
                    ])
                    for category, group in df.groupby('category'):
                        safe_cat = re.sub(r'[\\/*?:"<>|]', '_', category)
                        filename = os.path.join(OUTPUT_DIR, f"{safe_cat}.csv")
                        header = not os.path.exists(filename)
                        group.to_csv(filename, mode='a', index=False, header=header)
                
                pbar.update(1)
    
    generate_validation_report()

if __name__ == "__main__":
    main()
    print("处理完成！输出位置：", BASE_DIR)
