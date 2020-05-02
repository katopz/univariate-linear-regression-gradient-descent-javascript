import * as math from 'mathjs'
import csvToMatrix from 'csv-to-array-matrix'

csvToMatrix('./src/data.csv', init, ',')

function init(matrix: any) {
  // Part 0: Preparation
  console.log('Part 0: Preparation ...\n')

  let X = math.evaluate('matrix[:, 1]', {
    matrix
  })
  let y = math.evaluate('matrix[:, 2]', {
    matrix
  })

  let m = y.length

  // Part 1: Cost
  console.log('Part 1: Cost ...\n')

  // Add Intercept Term
  X = math.concat(math.ones([m, 1]).valueOf() as any, X)

  let theta = [[-1], [2]]
  let J = computeCost(X, y, theta)

  console.log('Cost: ', J)
  console.log('with: ', theta)
  console.log('\n')

  theta = [[0], [0]]
  J = computeCost(X, y, theta)

  console.log('Cost: ', J)
  console.log('with: ', theta)
  console.log('\n')

  // Part 2: Gradient Descent
  console.log('Part 2: Gradient Descent ...\n')

  const ITERATIONS = 1500
  const ALPHA = 0.01

  theta = gradientDescent(X, y, theta, ALPHA, ITERATIONS)

  console.log('theta: ', theta)
}

function computeCost(X: number, y: number[], theta: number[][]) {
  let m = y.length

  let predictions = math.evaluate('X * theta', {
    X,
    theta
  })

  let sqrErrors = math.evaluate('(predictions - y).^2', {
    predictions,
    y
  })

  let J = math.evaluate(`1 / (2 * m) * sum(sqrErrors)`, {
    m,
    sqrErrors
  })

  return J
}

function gradientDescent(X: number, y: number[], theta: number[][], ALPHA: number, ITERATIONS: number) {
  let m = y.length

  let thetaZero = theta[0]
  let thetaOne = theta[1]

  for (let i = 0; i < ITERATIONS; i++) {
    let predictions = math.evaluate('X * theta', {
      X,
      theta: [thetaZero, thetaOne]
    })

    thetaZero = math.evaluate(`thetaZero - ALPHA * (1 / m) * sum(predictions - y)`, {
      thetaZero,
      ALPHA,
      m,
      predictions,
      y
    })

    thetaOne = math.evaluate(`thetaOne - ALPHA * (1 / m) * sum((predictions - y)' * X[:, 2])`, {
      thetaOne,
      ALPHA,
      m,
      predictions,
      y,
      X
    })
  }

  return [thetaZero, thetaOne]
}

// @ts-ignore
function gradientDescentAlternative(X: number, y: number[], theta: number[][], ALPHA: number, ITERATIONS: number) {
  let m = y.length

  let thetaZero = theta[0]
  let thetaOne = theta[1]

  for (let i = 0; i < ITERATIONS; i++) {
    let predictions = math.evaluate('X * theta', {
      X,
      theta: [thetaZero, thetaOne]
    })

    thetaZero = math.evaluate(`thetaZero - ALPHA * (1 / m) * sum((predictions - y) .* X[:, 1])`, {
      thetaZero,
      ALPHA,
      m,
      predictions,
      y,
      X
    })

    thetaOne = math.evaluate(`thetaOne - ALPHA * (1 / m) * sum((predictions - y) .* X[:, 2])`, {
      thetaOne,
      ALPHA,
      m,
      predictions,
      y,
      X
    })
  }

  return [thetaZero, thetaOne]
}
