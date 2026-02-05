import streamlit as st
import re
import urllib.request
import urllib.parse
import requests as rq
from bs4 import BeautifulSoup
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import platform
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 상수
STOPWORDS_FILE = './resources/user_stopwords.json'
TOKENIZER_FILE = './resources/my_tokenizer_hybrid.joblib'
RESOURCES_DIR = './resources'
MAX_ARTICLES = 1000
RATE_LIMIT_DELAY = 0.1

os.makedirs(RESOURCES_DIR, exist_ok=True)

@st.cache_resource
def load_tokenizer() -> Optional[object]:
    if os.path.exists(TOKENIZER_FILE):
        try:
            return joblib.load(TOKENIZER_FILE)
        except Exception as e:
            logger.error(f"토크나이저 로드 실패: {e}")
    return None

def load_user_stopwords() -> set:
    if os.path.exists(STOPWORDS_FILE):
        try:
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            pass
    return set()

def save_user_stopwords(stopwords_set: set):
    with open(STOPWORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(sorted(stopwords_set)), f, ensure_ascii=False)

def get_font_path() -> Optional[str]:
    system = platform.system()
    if system == 'Windows':
        return 'C:/Windows/Fonts/malgun.ttf'
    elif system == 'Darwin':
        return '/System/Library/Fonts/AppleGothic.ttf'
    elif system == 'Linux':
        return '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    return None

@st.cache_data(ttl=3600)
def get_naver_news(_client_id: str, _client_secret: str, keyword: str, 
                  display: int = 100, start: int = 1) -> List[dict]:
    enc_text = urllib.parse.quote(keyword)
    url = f"https://openapi.naver.com/v1/search/news.json?query={enc_text}&display={display}&start={start}"
    
    headers = {
        "X-Naver-Client-Id": _client_id,
        "X-Naver-Client-Secret": _client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; NewsAnalyzer/1.0)"
    }
    
    try:
        time.sleep(RATE_LIMIT_DELAY)
        response = rq.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('items', [])
    except Exception as e:
        logger.error(f"Naver API 오류: {e}")
        return []

def clean_text(text: str) -> str:
    text = re.sub(r'[<>\'\"&]', ' ', text)
    text = re.sub(r'\d+[년월일가요]', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_keywords_from_article(link: str, tokenizer, stopwords: set, 
                                n_keywords: int = 50) -> Dict[str, int]:
    if 'n.news.naver.com' not in link:
        return {}
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        res = rq.get(link, headers=headers, timeout=10)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        news_tag = soup.select_one('#dic_area') or soup.select_one('.news_end')
        
        if news_tag:
            text = clean_text(news_tag.get_text())
            if tokenizer:
                tokens = tokenizer.tokenize(text)
            else:
                tokens = [t for t in re.findall(r'\b\w{2,10}\b', text)]
            
            words = [t for t in tokens if 2 <= len(t) <= 10 and t not in stopwords]
            if words:
                counts = pd.Series(words).value_counts()
                return dict(counts.head(n_keywords))
    except:
        pass
    return {}

@st.cache_data
def generate_wordcloud(counts: Dict[str, int], _mask_path: str = None) -> plt.Figure:
    font_path = get_font_path()
    
    mask_img = None
    if _mask_path and os.path.exists(_mask_path):
        try:
            mask_img = np.array(Image.open(_mask_path))
        except:
            pass
    
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        width=1200, height=800,
        max_words=200,
        mask=mask_img,
        colormap='viridis'
    )
    wc.generate_from_frequencies(counts)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

def load_lottie(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        lottie_path = f"{RESOURCES_DIR}/header_logo.json"
        lottie = load_lottie(lottie_path)
        if lottie:
            st_lottie(lottie, height=100, key="logo")
        else:
            st.markdown("🔍")
    with col2:
        st.markdown("<h1 style='text-align:center; color:#1f77b4;'>뉴스 키워드 분석 Pro</h1>", unsafe_allow_html=True)
        st.caption("Naver 뉴스 + SoyNLP + 실시간 워드클라우드")
    with col3:
        if st.button("🔓 로그아웃", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 설정")
        st.info("**소용팔 사용자 맞춤**\n- SoyNLP 하이브리드\n- Rate Limit 대응\n- Joblib 캐싱")
        
        if st.button("🧠 토크나이저 테스트"):
            tokenizer = load_tokenizer()
            if tokenizer:
                st.success("✅ 로드됨")
                test_text = "클라우드풀스택개발자AWS"
                tokens = tokenizer.tokenize(test_text)
                st.write(f"`{test_text}` → {tokens}")
            else:
                st.error("❌ `./resources/my_tokenizer_hybrid.joblib` 필요")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

    render_sidebar()

    if not st.session_state.logged_in:
        st.title("🔑 Naver API 인증")
        col1, col2 = st.columns(2)
        with col1:
            client_id = st.text_input("Client ID", key="id_input")
        with col2:
            client_secret = st.text_input("Client Secret", type="password", key="secret_input")
        
        if st.button("✅ 시작", type="primary"):
            if client_id and client_secret:
                st.session_state.update({
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'logged_in': True
                })
                st.rerun()
            else:
                st.error("ID/Secret 입력하세요.")
        return

    render_header()
    
    # 불용어 관리
    with st.expander("🚫 불용어 관리"):
        stopwords = load_user_stopwords()
        st.info(f"저장됨: {len(stopwords)}개")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            to_remove = st.multiselect("삭제", options=sorted(stopwords))
        with col2:
            if st.button("🗑️ 삭제"):
                save_user_stopwords(stopwords - set(to_remove))
                st.rerun()
    
    # 검색 폼
    with st.form("search_form"):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            keyword = st.text_input("🔍 키워드")
        with col2:
            max_articles = st.slider("📄 기사수", 100, MAX_ARTICLES, 500)
        with col3:
            top_n = st.slider("⭐ 키워드수", 30, 200, 100)
        
        analyze = st.form_submit_button("🚀 분석", type="primary")
    
    if analyze and keyword:
        with st.spinner(f"'{keyword}' 분석..."):
            tokenizer = load_tokenizer()
            client_id = st.session_state.client_id
            client_secret = st.session_state.client_secret
            stopwords = load_user_stopwords()
            
            all_items = []
            progress_bar = st.progress(0)
            
            for page_start in range(1, max_articles + 1, 100):
                items = get_naver_news(client_id, client_secret, keyword, 100, page_start)
                all_items.extend(items[:max_articles - len(all_items)])
                progress = min(len(all_items) / max_articles, 1.0)
                progress_bar.progress(progress)
                if len(all_items) >= max_articles:
                    break
            
            if not all_items:
                st.error("기사 없음")
                return
            
            news_df = pd.DataFrame([{
                'date': item.get('pubDate', ''),
                'title': clean_text(item.get('title', '')),
                'link': item.get('link', '')
            } for item in all_items])
            
            total_stats = {}
            analysis_progress = st.progress(0)
            
            for idx, row in news_df.iterrows():
                article_keywords = extract_keywords_from_article(
                    row['link'], tokenizer, stopwords, top_n
                )
                for word, freq in article_keywords.items():
                    if word not in total_stats:
                        total_stats[word] = [0, 0]
                    total_stats[word][0] += 1
                    total_stats[word][1] += freq
                analysis_progress.progress((idx + 1) / len(news_df))
            
            sorted_stats = dict(sorted(
                total_stats.items(),
                key=lambda x: (x[1][0], x[1][1]), reverse=True
            ))
            
            st.session_state.analysis_results = {
                'stats': sorted_stats,
                'news_df': news_df,
                'keyword': keyword,
                'top_n': top_n
            }
            st.success(f"✅ 완료! {len(sorted_stats)} 키워드")
            st.rerun()
    
    # 결과
    if st.session_state.get('analysis_results'):
        results = st.session_state.analysis_results
        stats = results['stats']
        news_df = results['news_df']
        keyword = results['keyword']
        top_n = results['top_n']
        
        st.header(f"📊 '{keyword}' 결과")
        
        top_stats = list(stats.items())[:top_n]
        df_stats = pd.DataFrame([{
            '단어': word, 
            '기사수': count[0], 
            '총빈도': count[1], 
            'TF/기사': f"{count[1]/count[0]:.1f}"
        } for word, count in top_stats])
        
        st.dataframe(df_stats, use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("✨ 워드클라우드")
            exclude_words = st.multiselect(
                "제외", list(stats.keys())[:top_n], key="wc_exclude"
            )
            if st.button("🎨 생성", type="primary"):
                wc_counts = {k: v[0] for k, v in stats.items() if k not in exclude_words}
                fig = generate_wordcloud(wc_counts)
                st.pyplot(fig)
                
                if exclude_words:
                    stopwords = load_user_stopwords()
                    save_user_stopwords(stopwords.union(set(exclude_words)))
                    st.toast("불용어 저장됨")
        
        with col2:
            st.subheader("📋 기사")
            st.dataframe(news_df.head(10), use_container_width=True)
        
        csv = df_stats.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📥 CSV",
            csv,
            f"{keyword}_keywords.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
