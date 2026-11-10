# Security and Responsible Handling

이 저장소는 jailbreak 평가와 유해 요청에 대한 모델 응답을 다룬다. 원시 프롬프트와 생성 응답은 오용 가능성과 데이터 라이선스 문제 때문에 Git 저장소에 커밋하지 않는다.

## 저장 가능

- 데이터셋 이름, revision, row ID, 해시
- 집계 지표와 범주별 통계
- 무해하게 편집한 오류 사례
- 실행 설정과 평가 코드

## 저장 금지

- API key, Hugging Face token, GitHub token
- 원시 jailbreak prompt 모음
- 실행 가능한 유해 절차를 포함한 모델 원문 응답
- 라이선스상 재배포할 수 없는 평가 데이터
- 개인 식별 정보

원문은 접근 제어된 artifact storage에 보관하고 run manifest에는 URI와 checksum만 기록한다. 외부 judge API에 보낼 수 있는 데이터 범위는 사용 전에 팀 책임자가 확인한다.

보안 문제나 credential 노출을 발견하면 공개 Issue를 만들지 말고 repository 관리자에게 비공개로 알린다. 노출된 credential은 즉시 폐기·교체하고 해당 commit은 단순 revert만으로 충분한지 별도 판단한다.
