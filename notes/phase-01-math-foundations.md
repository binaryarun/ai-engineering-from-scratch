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
(1.03 onwards to be added as I finish each lesson)

## Lesson 1.02 Vectors, Matrices & Operations

### Big idea
A matrix is a grid of numbers. In AI it is a set of detectors: each row looks at the input and produces one number. Stacking matrices (with a bit of non-linearity) gives a neural network.

### Basic operations
- Add / subtract: match positions, same shapes only.
- Multiply by a number (`A * 3`): every entry times 3.
- Element-wise (`A * B`): multiply matching positions. Example: `[[1,2],[3,4]] * [[5,6],[7,8]] = [[5,12],[21,32]]`.
- Matrix multiply (`A @ B`): row times column, then add. Same example gives `[[19,22],[43,50]]`. These are NOT the same thing: `*` is position by position, `@` is row-by-column.
- Shape rule: `(m x n) @ (n x p) = (m x p)`. The two middle numbers must match; the outer two are the result.

### Transpose (`A^T`)
- Rows become columns. Shape flips from `m x n` to `n x m`.
- Diagonal stays; the other numbers mirror across it.
- `[[4,7],[2,6]]^T = [[4,2],[7,6]]`. `2x3` becomes `3x2`.

### Inverse (`A^-1`)
- Meaning: the "undo" matrix. Like `1/5` undoes `5`. `A @ A^-1 = I` (identity).
- The `-1` is just part of the name. Nothing is multiplied by -1.
- 2x2 recipe for `[[a,b],[c,d]]`:
  1. det = `ad - bc`
  2. Swap the diagonal numbers (`a` and `d`)
  3. Flip the sign of `b` and `c` but leave them in their own spots
  4. Divide every entry by det
- Worked example: `[[4,7],[2,6]]`: det = 24 - 14 = 10; swapped = `[[6,-7],[-2,4]]`; divide by 10 gives `[[0.6,-0.7],[-0.2,0.4]]`. Check: `A @ A^-1 = [[1,0],[0,1]]`.
- Common slip: putting `-b` and `-c` in swapped spots (that gives the transpose of the answer).
- Transpose vs inverse: transpose keeps the diagonal and swaps the others; inverse swaps the diagonal and keeps the others (with signs flipped).
- det = 0 means no inverse (singular matrix).
- Bigger matrices: transpose works the same at any size. Inverse only exists for square matrices, and the 2x2 shortcut does not extend; use row reduction by hand or a computer function. Diagonal-only matrix: just flip each diagonal number to `1/number`. In AI we rarely compute inverses directly.

### Neural network layer: the beach example
Question: go to the beach? Inputs (0 to 1): sunny, windy, crowded. Two judges (layer 1) and one decision-maker (layer 2).
- Judge A "nice weather": weights sunny +1, windy -1, crowded 0, bias 0.
- Judge B "peace and quiet": weights crowded -1, others 0, bias +0.5.
- ReLU: if a judge's number is negative, the judge goes silent (0). Positive stays.
- Decision-maker: `1 x A + 0.5 x B`.
- Every layer repeats three moves: multiply by weights, add bias, apply ReLU.

| Day | A | B | After ReLU | Score |
|---|---|---|---|---|
| sunny 0.9, windy 0.2, crowded 0.8 | 0.7 | -0.3 | 0.7, 0 | 0.7 |
| sunny 0.9, windy 0.2, crowded 0.1 | 0.7 | 0.4 | 0.7, 0.4 | 0.9 |
| sunny 0.5, windy 0.6, crowded 0.3 | -0.1 | 0.2 | 0, 0.2 | 0.1 (stay home) |

Mistakes I made (watch for these):
- Adding the raw inputs instead of multiplying each by its weight first.
- Forgetting a negative weight means "more of this makes the judge unhappier".
- Using the number before ReLU in the next layer; the next layer only sees the after-ReLU numbers.
- Adding a weight (0.5) instead of multiplying by it.

### Reading the `matrices.py` output
- Weight matrix demo: `W = [[1,0,0],[0,1,0],[0.5,0.5,0]]`, `x = [0.8,0.6,0.1]` gives `[0.8, 0.6, 0.7]`. Row 0 copies feature 0, row 1 copies feature 1, row 2 averages them, and the third input is ignored (zero column).
- Forward pass shapes: `x (3,1)`, `W1 (4,3)`, `W2 (2,4)`. Layer 1: `(4x3)@(3x1)+(4x1) -> (4x1) -> ReLU`. Layer 2: `(2x4)@(4x1)+(2x1) -> (2x1)`.
- Only 1 of 4 hidden values survived ReLU (`0.1722`). Output `[-0.1037, 0.0308]` is meaningless because the weights are random and untrained.

### Practice still open
- Inverse of `[[3,1],[5,2]]` (det = 1).
- Transpose of `[[1,2,3],[4,5,6]]` (should be 3x2).

## Lesson 1.03 Matrix Transformations (partly done: chunks 1 and 2)

### Big idea
A matrix is a machine that moves every point. Its **columns say where the two basic steps land**: column 1 = where "one step right" `(1, 0)` lands, column 2 = where "one step up" `(0, 1)` lands. Any point is "some right steps + some up steps", so once you know those two landing spots you know where everything goes.

### Real-world picture: the tea shop
- Recipe grid (columns = drinks, rows = milk and sugar): `[[100, 150], [2, 1]]`. One tea = 100 ml milk + 2 spoons sugar; one coffee = 150 ml milk + 1 spoon sugar.
- Order 3 teas + 2 coffees = 3 x column 1 + 2 x column 2 = (600 ml, 8 spoons). That is exactly `recipe @ [3, 2]`.
- Order 4 teas + 1 coffee = (550 ml, 9 spoons). (I wrongly wrote coffee as 200 ml once: it is 150 ml.)
- Neural network layer is the same: each column = how much one input pushes every neuron; each row = one neuron's weights over all inputs.

### Common transformations (2x2)
| Name | Matrix | What it does |
|---|---|---|
| Stretch / scale | `[[2,0],[0,3]]` | x longer by 2, y longer by 3. `(1,1)` goes to `(2,3)` |
| Shear (lean) | `[[1,1],[0,1]]` | up arrow leans right; italic text. `(1,1)` goes to `(2,1)` |
| Mirror | `[[-1,0],[0,1]]` | selfie flip. `(2,1)` goes to `(-2,1)` |
| Rotate 90 | `[[0,-1],[1,0]]` | right arrow becomes up arrow. `(1,0)` goes to `(0,1)` |
Phone examples: selfie flip = mirror, widescreen stretch = scale, italic = shear, rotate photo = rotation.

### Chaining machines (composition)
- Tea shop chain: recipe grid then price list (milk Rs 0.05/ml, sugar Rs 1/spoon). 3 teas + 2 coffees = 600 ml + 8 spoons, then Rs 38.
- Merge the two machines into one: `price @ recipe = [7, 8.5]` (one tea Rs 7, one coffee Rs 8.5). 3 x 7 + 2 x 8.5 = Rs 38. Same answer.
- The middle shape numbers must match because machine 1's output count must equal machine 2's input count.
- `B @ A @ x` reads right to left: A acts first, then B (socks then shoes).
- Order matters: rotate 90 then scale (2, 0.5) sends `(1,0)` to `(0, 0.5)`; scale then rotate sends it to `(0, 2)`. `A @ B` is usually not `B @ A`.
- AI link: without a non-linearity like ReLU, a chain of layers collapses into one matrix. ReLU between layers stops that.

### Still to do for 1.03 (come back after the first neural-net exercise)
- Chunk 3: eigenvectors and eigenvalues (the special direction a matrix only stretches, never turns), PCA/stability link.
- Determinant as area scale factor (rotation 1, scale = product, shear 1, mirror -1, 0 = squashed flat).
- Run `phases/01-math-foundations/03-matrix-transformations/code/transformations.py` and the 3 exercises.
- Open check: 4 teas + 1 coffee money total both ways (answer Rs 36.5).
