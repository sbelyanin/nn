package nn

import (
	"math"
	"math/rand"
	"time"
)

// Сигмоидальная функция активации
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Производная сигмоидальной функции
func sigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// Инициализация весов случайными значениями
func randomMatrix(rows, cols int) [][]float64 {
	// Создаем новый источник с текущим временем в качестве seed
	src := rand.NewSource(time.Now().UnixNano())
	// Создаем локальный генератор на основе источника
	r := rand.New(src)
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			matrix[i][j] = r.NormFloat64()
		}
	}
	return matrix
}

// Умножение матриц
func dotProduct(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	//fmt.Println(len(a), len(a[0]), len(b), len(b[0]))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
		for j := range result[i] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
				//fmt.Println(i, j, "= a", i, k, "b=", k, j)
			}
		}
		//fmt.Println()
	}
	return result
}

// Транспонирование матрицы
func transpose(matrix [][]float64) [][]float64 {
	result := make([][]float64, len(matrix[0]))
	for i := range result {
		result[i] = make([]float64, len(matrix))
		for j := range result[i] {
			result[i][j] = matrix[j][i]
		}
	}
	return result
}

// Поэлементное умножение матриц
func elementwiseMultiply(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[i]))
		for j := range result[i] {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result
}

// Нейронная сеть
type NeuralNetwork struct {
	inputLayerSize      int
	hiddenLayerSize     int
	outputLayerSize     int
	weightsInputHidden  [][]float64
	weightsHiddenOutput [][]float64
	learningRate        float64
}

// Инициализация нейронной сети
func NewNeuralNetwork(input, hidden, output int, learningRate float64) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputLayerSize:  input,
		hiddenLayerSize: hidden,
		outputLayerSize: output,
		learningRate:    learningRate,
	}
	nn.weightsInputHidden = randomMatrix(input, hidden)
	nn.weightsHiddenOutput = randomMatrix(hidden, output)
	return nn
}

// Прямое распространение (feedforward)
func (nn *NeuralNetwork) feedforward(input [][]float64) [][]float64 {
	hiddenInputs := dotProduct(input, nn.weightsInputHidden)
	hiddenOutputs := applyActivation(hiddenInputs, sigmoid)
	outputInputs := dotProduct(hiddenOutputs, nn.weightsHiddenOutput)
	outputOutputs := applyActivation(outputInputs, sigmoid)
	return outputOutputs
}

// Обратное распространение ошибки (backpropagation)
func (nn *NeuralNetwork) train(input, target [][]float64) {
	// Прямое распространение
	hiddenInputs := dotProduct(input, nn.weightsInputHidden)
	hiddenOutputs := applyActivation(hiddenInputs, sigmoid)
	outputInputs := dotProduct(hiddenOutputs, nn.weightsHiddenOutput)
	outputOutputs := applyActivation(outputInputs, sigmoid)

	// Вычисление ошибки на выходном слое
	outputErrors := subtractMatrix(target, outputOutputs)
	outputGradients := applyActivationDerivative(outputOutputs, sigmoidDerivative)
	outputGradients = elementwiseMultiply(outputGradients, outputErrors)
	outputGradients = scalarMultiply(outputGradients, nn.learningRate)

	// Обновление весов между скрытым и выходным слоем
	hiddenOutputsT := transpose(hiddenOutputs)
	weightsHiddenOutputDeltas := dotProduct(hiddenOutputsT, outputGradients)
	nn.weightsHiddenOutput = addMatrix(nn.weightsHiddenOutput, weightsHiddenOutputDeltas)

	// Вычисление ошибки на скрытом слое
	weightsHiddenOutputT := transpose(nn.weightsHiddenOutput)
	hiddenErrors := dotProduct(outputErrors, weightsHiddenOutputT)
	hiddenGradients := applyActivationDerivative(hiddenOutputs, sigmoidDerivative)
	hiddenGradients = elementwiseMultiply(hiddenGradients, hiddenErrors)
	hiddenGradients = scalarMultiply(hiddenGradients, nn.learningRate)

	// Обновление весов между входным и скрытым слоем
	inputT := transpose(input)
	weightsInputHiddenDeltas := dotProduct(inputT, hiddenGradients)
	nn.weightsInputHidden = addMatrix(nn.weightsInputHidden, weightsInputHiddenDeltas)
}

// Применение функции активации к матрице
func applyActivation(matrix [][]float64, activation func(float64) float64) [][]float64 {
	result := make([][]float64, len(matrix))
	for i := range result {
		result[i] = make([]float64, len(matrix[i]))
		for j := range result[i] {
			result[i][j] = activation(matrix[i][j])
		}
	}
	return result
}

// Применение производной функции активации к матрице
func applyActivationDerivative(matrix [][]float64, derivative func(float64) float64) [][]float64 {
	result := make([][]float64, len(matrix))
	for i := range result {
		result[i] = make([]float64, len(matrix[i]))
		for j := range result[i] {
			result[i][j] = derivative(matrix[i][j])
		}
	}
	return result
}

// Вычитание матриц
func subtractMatrix(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[i]))
		for j := range result[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

// Умножение матрицы на скаляр
func scalarMultiply(matrix [][]float64, scalar float64) [][]float64 {
	result := make([][]float64, len(matrix))
	for i := range result {
		result[i] = make([]float64, len(matrix[i]))
		for j := range result[i] {
			result[i][j] = matrix[i][j] * scalar
		}
	}
	return result
}

// Сложение матриц
func addMatrix(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range result {
		result[i] = make([]float64, len(a[i]))
		for j := range result[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

