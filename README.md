# A-Federated-Learning-Method-of-Variational-Autoencoders-for-Collaborative-Filtering-

Collaborative Filtering(CF)은 사용자들의 선호도 유사성을 분석하여 사용자의 선호도를 예측 및 추천 하는 방법이다. Artificial Intelligent(AI) 이전에는 행렬분해(Matrix Factorization)기법을 사용하여 CF문제를 해결하는 연구가 많이 진행되었으나 최근 AI의 발전으로 CF문제를 딥 러닝으로 접근하려는 시도가 굉장 히 많아졌다. 특히 Variational Autoencoders(VAE)는 CF문제에 효과적으로 접근하여 좋은 성능을 내고 있 다. 하지만 VAE를 학습시키려면 사용자들의 해당 컨텐츠(Contents)에 대한 선호도 데이터가 있어야한다. 선호도 데이터는 사용자의 자발적인 참여로 생성되기 때문에 생산적이 측면에서 굉장히 비효율적이다. 또 한 AI의 예측 및 알고리즘의 사용자 감정분석을 통한 선호도 예측 데이터는 프라이버시(privacy)에 굉장히 민감하게 된다. 이에 본 논문은 사용자의 프라이버시를 완벽히 보존하면서 VAEs를 학습시키는 연합학습 (Federated Learning)방법을 설명한다. 또한 사용자, 엣지(edge), 중앙서버 모두가 학습에 참여하는 이 방 법이 선호도 데이터 특성상 기존 연합학습 방법이 학습시키지 못하는 VAE를 효과적으로 학습시킬 수 있 다는 것을 설명함과 동시에 효율적인 연합학습 방법을 추가로 제시한다.

- 시작 : python kscGraph.py
