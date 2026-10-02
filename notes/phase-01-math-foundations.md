# Phase 1: Math Foundations: my revision notes

Status: in progress. Done so far: 1.01. Next: 1.02 Vectors, Matrices and Operations.
Style of these notes: plain English, small numbers, one picture per idea.

## 1.01 Linear Algebra Intuition

### Vector = a list of numbers = walking instructions
`[3, 4]` means "3 steps east, 4 steps north". In AI a word is a vector of ~768 numbers, and an image is a long vector of pixel brightnesses.

### Length of a vector
Square each number, add them, take the square root.
- `[3, 4]` -> `sqrt(9 + 16)` = `sqrt(25)` = **5**
- `[1, 2, 3]` -> `sqrt(1 + 4 + 9)` = `sqrt(14)` = **3.7417**
- `[5, 12]` -> `sqrt(25 + 144)` = **13**

Why 13 and not 12? Length is the straight-line shortcut across the field. It is longer than either leg (12) but shorter than walking both legs (5 + 12 = 17).

### Normalising = shrink to length 1, keep the direction
Divide every number by the length.
- `[3, 4] / 5` = `[0.6, 0.8]`
- `[6, 8] / 10` = `[0.6, 0.8]` (same road, just twice as far)
- Check: `0.6^2 + 0.8^2 = 0.36 + 0.64 = 1`

Use it when you want to compare direction and ignore size.

### Dot product `a . b` = similarity
Multiply matching numbers, add up: `[1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32`.
- positive: pointing the same way. Zero: perpendicular (unrelated). Negative: opposite.
- Search, recommendations and RAG all rank things by this.

### Cosine similarity = dot product / (length a x length b)
Dividing by the lengths removes size, so only direction counts. 1 = same direction, 0 = unrelated, -1 = opposite. (My run: 0.9746, almost the same direction.)
- Example: `[1, 0]` and `[1, 1]`: `1 / (1 x 1.414) = 0.707`, which is the cosine of 45 degrees.

### Matrix = a transformation (it moves vectors)
Matrix x vector gives a new vector. Example: a rotation matrix turned `[3, 1]` by 90 degrees into `[-1, 3]`.

### Angle between vectors
`[1,0]` vs `[0,1]` = 90 degrees. `[1,0]` vs `[1,1]` = 45 degrees. A vector with itself = 0 degrees. Dot product = 0 exactly when the angle is 90.

### Projection = the shadow
The part of `a` that lies along `b`.
```
amount = (a . b) / (b . b)
shadow = amount x b
```
- `a = [3, 4]`, `b = [1, 0]`: shadow `[3, 0]`, leftover (residual) `[0, 4]`.
- Shadow + residual = `a`. The residual is perpendicular to `b` (`residual . b = 0`).
- Only the direction of `b` matters, not its length.
- Used in: least-squares line fitting, PCA, attention.
If this still feels fuzzy, remember only: **projection = the part of one arrow that lies along another**.

### Linear independence = "does this arrow add anything new?"
- East, north and up (`e1, e2, e3`): each opens a new dimension -> independent.
- `{e1, e2, 2*e1 + e2}`: the third is just a mix of the first two -> dependent (it can't take you up).
- In data: a "size in sq m" column that is just "size in sq ft" x 0.093 adds nothing and makes weights unstable.
- **Rank** = the number of truly new directions. `[[1,2],[2,4]]` has rank 1, because row 2 is row 1 doubled.

### Gram-Schmidt = tidy crooked arrows into perpendicular unit arrows
1. Normalise the first arrow -> `u1`.
2. Take the second, subtract its shadow on `u1`, normalise -> `u2`.
3. Take the third, subtract its shadows on `u1` and `u2`, normalise -> `u3`.
Check the result: every `|u| = 1` and every `u_i . u_j = 0`.
- Tiny example: `v1 = [3, 0]`, `v2 = [2, 2]` -> `u1 = [1, 0]`; shadow of `v2` on it is `[2, 0]`; leftover `[0, 2]` -> `u2 = [0, 1]`.
- Rarely written by hand. It sits inside QR decomposition, PCA and SVD.

### The neural network layer = matrix x vector (+ bias, + activation)
Each row of the weight matrix is one worker's **recipe**: which inputs to care about, and how much. The layer is `ReLU(W @ x + b)`.

Tiny worked example (inputs size, rooms, age = `[2, 3, 1]`):
```
Layer 1:  W1 = [[0.5, 1.0, -0.5],   bias [-4, 0]
                [0.0, 0.2,  0.1]]
  recipes:  [3.5, 0.7]    + bias -> [-0.5, 0.7]    ReLU -> [0, 0.7]
Layer 2:  W2 = [[2, 10]],  bias 1
  2*0 + 10*0.7 + 1 = 8
```
- ReLU = "if negative, say 0". Without it, stacked layers collapse into one matrix.
- Layer 2 reads layer 1's report, not the original input.

Real example in plain English, a digit recogniser:
- The input is 784 brightness numbers (28 x 28 dots).
- Layer 1 = detectors: "top line?", "slanting line?", "closed loop?", "vertical line?".
- Layer 2 combines them: top line + slanting line + no loop = **7**.
- Nobody writes those detectors. Weights start random, the network answers badly, training nudges every number to be less wrong, repeated over thousands of labelled digits.
- In a real model the knowledge lives in billions of numbers in these matrices.

One-sentence answer: *"A matrix-vector product mixes the input numbers using learned weights to produce a new vector, which is exactly what one layer of a neural network does."*

### My `vectors.py` output, decoded
- `|a| = 3.7417`: length of `[1, 2, 3]`.
- `cosine_similarity = 0.9746`: very similar direction.
- `Rotate [3,1] by 90 -> [-1,3]`: a matrix moving a point.
- `proj = [3,0]`, `residual = [0,4]`, `residual . b = 0`: the shadow split.
- `independent: True / False`: whether arrows add something new.
- `u1.u2 = 0`, `|u| = 1`: Gram-Schmidt worked.
- `ranks 2, 1, 2`: number of independent directions.
- The final layer output `[-0.0197, 0.1087]` looks random because the weights are random and untrained.

### Quick self-test
1. Length of `[6, 8]`? (10)
2. Normalise `[5, 12]`? (about `[0.38, 0.92]`)
3. `[1, 2, 3] . [4, 5, 6]`? (32)
4. Shadow of `[5, 3]` on `[1, 0]`? (`[5, 0]`; residual `[0, 3]`)
5. Layer 1 and 2 above with `x = [1, 0, 2]`? (layer 1: `[-4.5, 0.2]` -> ReLU `[0, 0.2]`; layer 2: 3)

## 1.02 onwards
(to be added as I finish each lesson)
