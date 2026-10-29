# 4pm 올인원 교재 시스템

고등 영어 내신 교재(한판승, Self-Note, Core, Voca, 기출 어휘)를 Gemini로 생성하고 HTML 단행본으로 출력하는 Streamlit 앱.

## 구조
- `app.py` 한 파일에 전부 들어 있음: 저장소(Supabase/JSON) → 교재별 CSS·렌더러 → Gemini 프롬프트·스키마 → 라이브러리 관리 UI → 탭 7개.
- 배포: GitHub `main`에 push → Streamlit Cloud 자동 재배포.
- 데이터: Supabase `app_storage` 테이블(`doc_key` = victory_library, anatomy_library, core_library, voca_library, past_voca_library, exam_dna_rules), 이미지는 Storage `image_assets` 버킷. Streamlit Cloud의 로컬 파일은 재시작 시 사라지므로 Supabase가 원본.

## 규칙
- 화면 문구는 한국어. 브랜드명(PQRS영쌤, 포피엠 입시 영어, Anatomy 시리즈명)은 바꾸지 않는다.
- 저장된 데이터 구조(`voca_list`, `summaries`, `analysis` 등 키 이름)를 바꿀 때는 기존 데이터를 읽어 변환하는 코드도 함께 넣는다.
- API 키·비밀번호는 `.streamlit/secrets.toml`(로컬) / Streamlit Cloud Secrets에만 둔다. 코드에 하드코딩 금지.
- 수정 후 `streamlit run app.py`로 로컬 확인, push는 사용자 확인 후에만.
