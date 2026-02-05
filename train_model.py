import os
import math
import logging
import logging.handlers
import urllib.request
import yaml
import sys
from typing import Dict, Any
import joblib
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

# ===== 로깅 설정 (여기부터!) =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 (stdout으로 확실히 출력)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# 파일 핸들러 (train.log 저장)
file_handler = logging.handlers.RotatingFileHandler(
    'train.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

logger.info("로깅 시스템 시작!")  # 테스트 로그
# ===== 로깅 끝 =====
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """YAML config 파일 로드 (강화 버전)"""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Config 로드: {config_path}")
                return config
        else:
            # 기본 config
            default_config = {
                'data_url': 'https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt',
                'data_file': 'data_train.txt',
                'model_dir': './resources',
                'extractor_params': {
                    'min_frequency': 100,
                    'min_cohesion_forward': 0.05,
                    'min_right_branching_entropy': 0.0
                },
                'tokenizer_types': ['cohesion', 'branching', 'hybrid']
            }
            logger.info("Config 파일 없음, 기본값 사용")
            return default_config
    except Exception as e:
        logger.error(f"Config 로드 실패: {e}")
        logger.info("기본 config 사용")
        return {  # 최소 기본값
            'data_url': 'https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt',
            'data_file': 'data_train.txt',
            'model_dir': './resources',
            'extractor_params': {'min_frequency': 100},
            'tokenizer_types': ['cohesion']
        }
def download_data(url: str, filename: str) -> None:
    """데이터 다운로드 with 에러 핸들링"""
    try:
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        urllib.request.urlretrieve(url, filename)
        logger.info(f"데이터 다운로드 완료: {filename}")
    except Exception as e:
        logger.error(f"데이터 다운로드 실패: {e}")
        raise

def train_extractor(corpus_file: str, params: Dict[str, Any]) -> WordExtractor:
    """WordExtractor 학습"""
    try:
        corpus = DoublespaceLineCorpus(corpus_file)
        logger.info(f"코퍼스 로드: {len(corpus)} 문서")
        
        extractor = WordExtractor(
            min_frequency=params['min_frequency'],
            min_cohesion_forward=params['min_cohesion_forward'],
            min_right_branching_entropy=params['min_right_branching_entropy']
        )
        extractor.train(corpus)
        logger.info("WordExtractor 학습 완료")
        return extractor
    except Exception as e:
        logger.error(f"학습 실패: {e}")
        raise

def create_scores(word_score_table: Dict[str, Any], score_type: str) -> Dict[str, float]:
    """다양한 스코어 계산"""
    scores = {}
    for word, score in word_score_table.items():
        if score_type == 'cohesion':
            scores[word] = score.cohesion_forward
        elif score_type == 'branching':
            scores[word] = score.right_branching_entropy
        elif score_type == 'hybrid':
            cohesion = score.cohesion_forward
            branching = math.exp(score.right_branching_entropy)  # 지수 가중치
            scores[word] = cohesion * branching
        else:
            raise ValueError(f"Unknown score_type: {score_type}")
    logger.info(f"{score_type} 스코어 테이블 생성: {len(scores)} 단어")
    return scores

def save_tokenizer(tokenizer: LTokenizer, filepath: str) -> None:
    """joblib으로 토크나이저 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(tokenizer, filepath)
    logger.info(f"토크나이저 저장: {filepath}")

def test_tokenizer(tokenizer: LTokenizer, test_sentences: list) -> None:
    """토크나이저 테스트"""
    logger.info("토크나이저 테스트:")
    for sent in test_sentences:
        tokens = tokenizer.tokenize(sent)
        logger.info(f"입력: {sent} -> 출력: {tokens}")

def main():
    config = load_config()
    
    # 1. 데이터 다운로드
    download_data(config['data_url'], config['data_file'])
    
    # 2. extractor 학습
    extractor = train_extractor(config['data_file'], config['extractor_params'])
    word_score_table = extractor.extract()
    
    # 3. 토크나이저 생성 및 저장
    test_sentences = ["한국어토크나이저테스트입니다.", "신조어및복합어처리가중요합니다."]
    
    for t_type in config['tokenizer_types']:
        scores = create_scores(word_score_table, t_type)
        tokenizer = LTokenizer(scores=scores)
        
        # 테스트
        test_tokenizer(tokenizer, test_sentences)
        
        # 저장
        filename = f"my_tokenizer_{t_type}.joblib"
        filepath = os.path.join(config['model_dir'], filename)
        save_tokenizer(tokenizer, filepath)

if __name__ == "__main__":
    main()
