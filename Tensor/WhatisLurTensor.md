LurTensor를 만든 이유 / 특징
----------------

Julia에는 MATLAB과 거의 비슷한 array 문법이 있습니다. 아마 MATLAB 코드의 5% 정도만 수정해도 julia에서 똑같이 돌아가게 할 수 있을 것이라고 생각하며, 처음에는 이 방식으로 모든 걸 해보고자 했습니다.

하지만 코드를 짤 때마다 텐서의 leg들이 어떤 순서로 배열되어있는지 기억하거나 필기해둬야 했고, 적어도 저는 이런 방식에 잘 적응이 되지 않았습니다. 따라서, 처음에는 제가 지금까지 썼던 익숙한 패키지인 ITensors를 이용하여 번역을 하고자 했습니다. 다만 ITensor를 사용하면 ITensor 내부 array에 직접 접근하는 것이 상당히 까다로워집니다. 예를 들면, Iterative diagonalization을 구현할 때 eigen decomposition의 결과를 degeneracy를 고려하여 truncation하는 부분(교수님 코드의 Tutorials/T05.1_IterativeDiag_sol/IterativeDiag_sol.m의 55 ~ 64번째 줄)이 있는데, ITensor를 사용하면 이 부분에서 코드가 매우 복잡해집니다. 


따라서 julia 공부도 할 겸, ITensor과는 다른 LurTensor라는 라이브러리를 새로 만들게 되었습니다. 이를 사용하면
1. 기존 코드와 달리, leg 순서를 기억할 필요가 없습니다.
2. ITensor과 달리, 기존의 인덱싱 방식 (ex- A[2:4, 3:5])으로 텐서의 일부분에 접근할 수 있습니다.

이름의 유래는 그냥 어디에나 붙이고 다니는 닉네임 (이메일에도 붙어있습니다) Lur + Tensor입니다. 제 이름을 걸고 만드는 것인 만큼 오류를 발견하셨다면 주저없이 제보 바랍니다.

LurTensor 사용 방법
------------------


LurTensor 튜토리얼
------------------
