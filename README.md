# CVFW   (Computer Vision Feature Weight)
-------------


각 픽셀마다 가중치를 구하고, 가중치의 분포를 분석하여 특징가중치를 골라낸다.  
이미지의 유사도를 체크할때에는 특징가중치의 유사도의 반영률이 제일 높다.



1. 가중치를 구한다.
- CVFW.Weight_()
- CVFW_GROUP.weight_all()
- 15 ~ 19  or 208 ~ 210  

2. 여러 가중치들에 대해서 특징 가중치를 구한다.
- COST_FUNCTION.feature_group()
- CVFW_GROUP.train_()
- 129 ~ 146 or 223 ~ 234  

3. 각 클래스마다 cost 를 측정하고 cost 가 가장 작은 클래스가 예측 클래스이다.
