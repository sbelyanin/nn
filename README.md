# nn
## Simple Neural Netwoek 

## Описание кода:
### Структура нейронной сети:
   NeuralNetwork содержит веса между входным и скрытым слоем (weightsInputHidden) и между скрытым и выходным слоем (weightsHiddenOutput).
   Также задается размерность каждого слоя и скорость обучения (learningRate).
### Функции активации:
   Используется сигмоидальная функция активации и ее производная.
### Обучение:
   В методе train реализован алгоритм обратного распространения ошибки (backpropagation), который обновляет веса сети на основе ошибки предсказания.
