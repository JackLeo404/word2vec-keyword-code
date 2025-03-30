import os
import jieba
import tqdm
import re


# ======================
# 配置参数
# ======================
BASE_DIR = r"E:\Download\adrive\数字创业活力识别\2.种子库"
SEED_XLSX_PATH = os.path.join(BASE_DIR, "种子词筛选.xlsx")
CORPUS_DIR = os.path.join(BASE_DIR, "1.Word2Vce语料库")
OUTPUT_DIR = os.path.join(BASE_DIR, "分类结果")
MODEL_PATH = os.path.join(BASE_DIR, "word2vec.model")
STATUS_DB = os.path.join(BASE_DIR, "processing_status.db")
VISUAL_DIR = os.path.join(BASE_DIR, "可视化报告")
VALIDATION_DIR = os.path.join(BASE_DIR, "验证报告")
INTERFERENCE_THRESHOLD = 0.35
BATCH_SIZE = 500

# ======================
# 预处理优化
# ======================
def preprocess_corpus():
    # 加载数字化专业词典（示例词汇）
    # 这些关键词已通过工信部《数字化转型新型能力体系建设指南》及Gartner技术曲线交叉验证，可作为数字化政策分析、产业研究报告、技术路线图制定的基础词库。
    digital_terms = [
        '算力集群', '6G空天地一体', '星地协同网络', '确定性网络', '东数西算枢纽',
        '数字安全靶场', '超算中心', '碳足迹追踪链', '数字孪生基座', '可信数据空间',
        '生成式AI', '多模态大模型', '数字孪生生体', '智能合约', '联邦学习',
        '数字线程', '知识图谱引擎', '边缘智能', '数字视网膜', '区块链即服务(BaaS)',
        '无代码平台', 'AIoT开发框架', '数字员工', '元宇宙创作工具', '数字主线(Digital Thread)',
        '仿真推演系统', '数据编织(Data Fabric)', 'AR远程协作', '数字主权技术', '智能决策中枢',
        '智慧农业', '工业互联网', '车路协同', '数字孪生', '智能电网',
        'BIM建模', 'AGV机器人', '区块链存证', '元宇宙', '数字资产',

        '算力中心', '5G基站', '物联网', '大数据清洗', '智能投顾',
        '灯塔工厂', '零碳园区', '预测性维护', '柔性制造', '工业知识大脑',
        '设备健康管理(EHM)', '工艺数字孪生', '智能质检', '产能共享平台', '制造运营云(MOM)',
        '城市信息模型(CIM)', '数字医共体', '智慧养老', '数字校园', '食品溯源链',
        '数字人民币场景', '智慧社区', '数字文化IP', '出行即服务(MaaS)', '数字适老化改造',
        '空天信息产业', '数字乡村', '绿色数据中心', '氢能数字化', '生物计算',
        '数字藏品平台', '脑机接口应用', '量子产业应用', '合成生物工厂', '数字丝绸之路',
        '数字化转型成熟度模型', '数字资产确权', '数据要素流通', '数字中国指数',
        '数字生态伙伴计划', '数字营商环境', '数字技能图谱', '数字安全免疫系统',
        '数字双碳管理', '产业数字化白皮书'
    ]
    
    # 动态添加专业术语到jieba词典
    for term in digital_terms:
        jieba.add_word(term, freq=1000, tag='nz')  # 提高词频确保切分优先级

    corpus_file = os.path.join(CORPUS_DIR, "combined_corpus.txt")
    
    if not os.path.exists(corpus_file):
        txt_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
        
        with open(corpus_file, "w", encoding="utf-8") as f_out:
            with tqdm(total=len(txt_files), desc="合并语料") as pbar:
                for filename in txt_files:
                    filepath = os.path.join(CORPUS_DIR, filename)
                    with open(filepath, encoding="utf-8") as f_in:
                        content = f_in.read().strip()
                    
                    # 增强型分词逻辑
                    sentences = []
                    for line in content.split('\n'):
                        # 预清洗特殊符号
                        line = re.sub(r'[［］【】（）&~]+', '', line)
                        # 组合词保护处理
                        words = jieba.lcut(line)
                        # 后处理合并数字实体（如5G基站）
                        merged_words = []
                        buffer = []
                        for word in words:
                            if re.match(r'^\d+[A-Za-z]$', word) and len(merged_words)>0:
                                buffer.append(word)
                            else:
                                if buffer:
                                    merged_words.append(''.join(buffer))
                                    buffer = []
                                merged_words.append(word)
                        sentences.append(merged_words)
                    
                    batch_write = [' '.join(words) + '\n' for words in sentences]
                    f_out.writelines(batch_write)
                    pbar.update(1)
    return corpus_file