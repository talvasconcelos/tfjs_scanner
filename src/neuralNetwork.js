import * as tf from '@tensorflow/tfjs';

class neuralNetwork {
  createAndCompileModel(layers, hiddenSize, rnnType = 'LSTM', timesteps, input_dimensions) {
    const model = tf.sequential()
    switch(rnnType) {
      case 'SimpleRNN':
        model.add(tf.layers.simpleRNN({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true,
          inputShape: [timesteps, input_dimensions]
        }))
        break;
      case 'LSTM':
        model.add(tf.layers.lstm({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true,
          inputShape: [timesteps, input_dimensions]
        }))
        break;
      case 'GRU':
        model.add(tf.layers.gru({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true,
          inputShape: [timesteps, input_dimensions]
        }))
        break;
      default:
        throw new Error(`Unsupported RNN type: '${rnnType}'`)
    }
    model.add(tf.layers.repeatVector({n: timesteps}))
    switch(rnnType) {
      case 'SimpleRNN':
        model.add(tf.layers.simpleRNN({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true
        }))
        break;
      case 'LSTM':
        model.add(tf.layers.lstm({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true
        }))
        break;
      case 'GRU':
        model.add(tf.layers.gru({
          units: hiddenSize,
          recurrentInitializer: 'glorotNormal',
          returnSequences: true
        }))
        break;
      default:
        throw new Error(`Unsupported RNN type: '${rnnType}'`)
    }
    model.add(tf.layers.dense({units: 1}))
    model.add(tf.layers.activation({activation: 'softmax'}))

    model.compile({
      loss: 'binaryCrossentropy',
      optimyzer: 'adam',
      metrics: ['accuracy']
    })

    return model
  }

  async train(trainData, validationData, iterations, batchSize, model) {
    const [trainXs, trainYs] = generateData(trainData)
    const [validXs, validYs] = generateData(validationData)
    for (var i = 0; i < iterations; i++) {
      const history = await model.fit(trainXs, trainYs, {
        epochs: 1,
        batchSize,
        validationData: [validXs, validYs]
      })
      const trainLoss = history.history['loss'][0]
      const trainAccuracy = history.history['acc'][0]
      const validLoss = history.history['val_loss'][0]
      const validAccuracy = history.history['val_acc'][0]
      console.log({trainLoss, trainAccuracy, validLoss, validAccuracy})
      await tf.nextFrame()
    }
  }

  predict() {
    tf.tidy(() => {
      const predictOut =
    })
  }
}

export default neuralNetwork
