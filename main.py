import streamlit as st
import re 
import urllib.request
import urllib.parse
import requests as rq
from bs4 import BeautifulSoup
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import platform
import os
import urllib.error
from typing import Dict, List, Optional, Tuple

# --- [1. 설정 및 데이터 관리 함수] ---
st.set_page_config(page_title="🔍 뉴스 키워드 시각화")

STOPWORDS_FILE = './resources/user_stopwords.json'

def load_user_stopwords():
    if os.path.exists(STOPWORDS_FILE):
        try:
            with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except: return set()
    return set()

def save_user_stopwords(stopwords_set):
    if not os.path.exists('./resources'):
        os.makedirs('./resources')
    with open(STOPWORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(sorted(stopwords_set)), f, ensure_ascii=False)

def get_naver_news(keyword, display, start):
    client_id = st.session_state.get('client_id')
    client_secret = st.session_state.get('client_secret')
    encText = urllib.parse.quote(keyword)
    url = f"https://openapi.naver.com/v1/search/news.json?query={encText}&display={display}&start={start}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    
    try:
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            return json.loads(response.read().decode('utf-8'))['items']
    except urllib.error.HTTPError as e:
        if e.code == 401:
            st.error("❌ API 키가 올바르지 않습니다. Client ID와 Secret을 다시 확인해주세요.")
        elif e.code == 403:
            st.error("❌ API 권한이 없거나 호출 한도를 초과했습니다.")
        else:
            st.error(f"❌ API 오류 발생: {e.code}")
        return None # 빈 리스트 대신 None을 반환하여 에러임을 표시
    except Exception as e:
        st.error(f"❌ 연결 오류: {str(e)}")
        return None

def cleanText(text):
    text = re.sub(r'\d|[a-zA-Z]|\W',' ', text)
    return re.sub(r'\s+',' ', text)

def cleanHtml(text):
    text = re.sub(r'<[^>]+>', '', text)
    for ent, char in [('&quot;', '"'), ('&apos;', "'"), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>')]:
        text = text.replace(ent, char)
    return text

@st.cache_resource
def getTokenizer():
    try:
        with open('./resources/my_tokenizer3.model','rb') as f:
            return pickle.load(f)
    except: return None

def get_font_path() -> Optional[str]:
    # 1. 먼저 resources 폴더에 직접 업로드한 폰트가 있는지 확인
    local_font = './resources/NanumGothic-Regular.ttf' # 파일명에 맞춰 수정
    if os.path.exists(local_font):
        return local_font
    
    # 2. 없을 경우 시스템 폰트 시도 (기존 로직)
    system = platform.system()
    if system == 'Windows':
        return 'C:/Windows/Fonts/malgun.ttf'
    elif system == 'Darwin':
        return '/System/Library/Fonts/AppleGothic.ttf'
    elif system == 'Linux':
        # 리눅스 서버 기본 폰트 경로 (설치되어 있을 경우)
        linux_font = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if os.path.exists(linux_font):
            return linux_font
    return None
def plotChart(count_dict, container):
    try:
        img_path = './resources/background_0.png'
        my_mask = np.array(Image.open(img_path)) if os.path.exists(img_path) else None
        wc = WordCloud(
            font_path=get_font_path(),
            background_color='white',
            width=500, height=500,
            max_words=300,
            mask=my_mask
        )
        # count_dict에는 이미 '기사 발생 수'가 값으로 들어있음
        wc.generate_from_frequencies(count_dict)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        container.pyplot(fig)
    except Exception as e:
        container.error(f"시각화 오류: {e}")

def load_lottie_local(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def render_header():
    col1, col2, col3 = st.columns([1, 4, 1], vertical_alignment="center")
    with col1:
        lottie_path = "./resources/header_logo.json"
        lottie_json = load_lottie_local(lottie_path)
        if lottie_json: st_lottie(lottie_json, speed=1, width=120, height=120, key="main_logo")
        else: st.markdown("### 🔍")
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔍 뉴스 키워드 시각화</h1>", unsafe_allow_html=True)
    with col3:
        if st.button("로그아웃", use_container_width=True):
            st.session_state.clear(); st.rerun()

# --- [3. 메인 로직] ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'analysis_step' not in st.session_state: st.session_state['analysis_step'] = False
if not st.session_state['logged_in']:
    st.title("🔑 Naver API 인증")
    
    col1, col2 = st.columns(2)
    with col1:
        c_id = st.text_input("Client ID")
    with col2:
        c_pw = st.text_input("Client Secret", type="password")
    
    if st.button("✅ 시작", use_container_width=True):
        if c_id and c_pw:
            # 임시로 세션에 저장해보고 테스트 호출
            st.session_state['client_id'] = c_id
            st.session_state['client_secret'] = c_pw
            
            # 테스트 검색 (단어 하나로 유효성 확인)
            test_res = get_naver_news("테스트", 1, 1)
            
            if test_res is not None: # 성공 시 (빈 리스트일지라도 None이 아니면 키는 정상)
                st.session_state['logged_in'] = True
                st.success("인증 성공!")
                st.rerun()
            else:
                # 에러 메시지는 get_naver_news 안에서 st.error로 출력됨
                st.session_state['client_id'] = None
                st.session_state['client_secret'] = None
        else:
            st.warning("ID와 Secret을 모두 입력해 주세요.")
else:
    render_header()
    
    with st.expander("🚫 불용어 관리"):
        saved_stops = load_user_stopwords()
        if saved_stops:
            st.write(f"총 {len(saved_stops)}개 저장됨")
            
            col1, col2= st.columns([4, 1])
            with col1: 
                
                to_del = st.multiselect(
                    "삭제할 단어 선택", # 내부적으로 필요한 이름
                    options=sorted(list(saved_stops)),
                    label_visibility="collapsed" # 레이블을 완전히 제거하고 공간도 안 차지함
                )
            with col2:
                if st.button("단어 삭제"):
                    save_user_stopwords(saved_stops - set(to_del)); st.rerun()
        else: st.info("저장된 단어가 없습니다.")

    with st.form(key='search_form'):
        
        sfcol1, sfcol2, sfcol3 = st.columns([3, 2, 2])
        with sfcol1:
            search_keyword = st.text_input("분석 키워드")
        with sfcol2: 
            m_amount = st.slider('수집 기사 수 (m)', 100, 800, 100, 100)
        with sfcol3:
            n_amount = st.slider('기사당 수집할 핵심 단어 수 (n)', 30, 300, 50)
        
        if st.form_submit_button("분석 시작"):
            if search_keyword:
                my_tokenizer = getTokenizer()
                items = []
                for i in range(m_amount // 100):
                    items.extend(get_naver_news(search_keyword, 100, (i*100)+1))
                
                if items:
                    total_stats = {} # {단어: [기사발생수, 총언급횟수]}
                    news_data_list = []
                    saved_stops = load_user_stopwords()
                    pbar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, item in enumerate(items):
                        news_data_list.append({'날짜': item['pubDate'], '제목': cleanHtml(item['title']), '링크': item['link']})
                        if 'n.news.naver.com' in item['link']:
                            try:
                                res = rq.get(item['link'], headers={'User-Agent':'Mozilla/5.0'}, timeout=5)
                                news_tag = BeautifulSoup(res.text, 'html.parser').select_one('#dic_area')
                                if news_tag:
                                    txt = cleanText(news_tag.text)
                                    tokens = [t[0] for t in my_tokenizer.tokenize(txt, flatten=False)]
                                    # 1. 단어 추출 및 단어장 필터링
                                    words = [t for t in tokens if 2 <= len(t) <= 10 and t not in saved_stops]
                                    if words:
                                        full_counts = pd.Series(words).value_counts()
                                        # 2. 상위 n_amount개 선정
                                        top_n = full_counts.head(n_amount)
                                        
                                        # 3. 듀얼 카운팅 (이진 가중치 + 실제 빈도)
                                        for word, count in top_n.items():
                                            if word not in total_stats:
                                                total_stats[word] = [0, 0]
                                            total_stats[word][0] += 1      # 기사 발생 수 (Binary)
                                            total_stats[word][1] += count  # 총 언급 횟수 (Raw Frequency)
                            except: continue
                        pbar.progress((idx + 1) / len(items))
                        status_text.text(f"기사 분석 중... ({idx+1}/{len(items)})")
                    
                    if total_stats:
                        # 4. 정렬: 1순위 기사수(x[1][0]), 2순위 총빈도(x[1][1])
                        sorted_stats = dict(sorted(total_stats.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True))
                        st.session_state.update({
                            'total_stats': sorted_stats,
                            'current_keyword': search_keyword,
                            'current_n': n_amount,
                            'news_items': news_data_list,
                            'analysis_step': True
                        })
                        if 'display_dict' in st.session_state: del st.session_state['display_dict']
                    else: st.error("결과가 없습니다.")

    if st.session_state.get('analysis_step') and 'total_stats' in st.session_state:
        full_dict = st.session_state['total_stats']
        display_limit = st.session_state.get('current_n', 50)
        top_words = list(full_dict.keys())[:display_limit]
        saved_stops = load_user_stopwords()

        st.divider()
        st.subheader(f"🛠️ '{st.session_state['current_keyword']}' 결과 정제")
        use_auto = st.toggle("💾 영구 제외 단어장 자동 적용", value=True)
        default_sel = [w for w in top_words if w not in saved_stops] if use_auto else top_words
        selected = st.multiselect("포함할 단어 선택:", options=top_words, default=default_sel)

        if st.button("✨ 워드클라우드 생성"):
            removed = set(top_words) - set(selected)
            if use_auto and removed:
                save_user_stopwords(saved_stops.union(removed))
                st.toast(f"{len(removed)}개 단어 저장됨")
            # 워드클라우드용으로는 '기사 발생 수'만 전달
            st.session_state['display_dict'] = {k: full_dict[k][0] for k in selected}
            st.rerun()

        if 'display_dict' in st.session_state:
            st.divider()
            c1, c2 = st.columns([2, 1.5])
            with c1: plotChart(st.session_state['display_dict'], st)
            with c2: 
                st.write("### 📈 기사수 & 총빈도")
                # 테이블에는 두 수치를 모두 표시하여 정렬 기준 확인 가능하게 함
                stat_data = [
                    {'단어': k, '등장 기사 수': full_dict[k][0], '총 언급 횟수': full_dict[k][1]} 
                    for k in st.session_state['display_dict'].keys()
                ]
                st.dataframe(pd.DataFrame(stat_data), use_container_width=True)

            # --- [추가된 섹션: 스크랩 기사 목록] ---
            st.divider()
            st.subheader("📰 분석된 기사 원문 목록")
            
            if 'news_items' in st.session_state and st.session_state['news_items']:
                # 데이터프레임으로 변환
                df_news = pd.DataFrame(st.session_state['news_items'])
                
                # 링크를 클릭 가능한 형태로 보여주기 (옵션)
                # 데이터프레임 내에서 링크를 직접 클릭하게 하려면 st.column_config를 사용합니다.
                st.dataframe(
                    df_news,
                    column_config={
                        "링크": st.column_config.LinkColumn("기사 링크"),
                        "날짜": st.column_config.DateColumn("발행일", format="YYYY-MM-DD")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("수집된 기사 정보가 없습니다.")