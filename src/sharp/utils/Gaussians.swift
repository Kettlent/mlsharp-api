import Foundation

/// Contains basic data structures and functionality for 3D Gaussians.
/// For licensing see accompanying LICENSE file.
/// Copyright (C) 2025 Apple Inc. All Rights Reserved.

// MARK: - Type Aliases

typealias BackgroundColor = String // "black", "white", "random_color", "random_pixel"

// MARK: - Data Structures

/// Represents a collection of 3D Gaussians.
struct Gaussians3D {
    var meanVectors: [[Float]] // N x 3
    var singularValues: [[Float]] // N x 3
    var quaternions: [[Float]] // N x 4
    var colors: [[Float]] // N x 3
    var opacities: [Float] // N
    
    mutating func to(device: String) -> Gaussians3D {
        // In Swift, device management is implicit; return self
        return self
    }
}

/// Meta data about Gaussian scene.
struct SceneMetaData {
    var focalLengthPx: Float
    var resolutionPx: (width: Int, height: Int)
    var colorSpace: String
}

// MARK: - Matrix Operations

/// Simple matrix representation
struct Matrix {
    var data: [Float]
    let rows: Int
    let cols: Int
    
    init(rows: Int, cols: Int, data: [Float]) {
        self.rows = rows
        self.cols = cols
        self.data = data
    }
    
    subscript(row: Int, col: Int) -> Float {
        get { data[row * cols + col] }
        set { data[row * cols + col] = newValue }
    }
    
    /// Transpose the matrix
    func transpose() -> Matrix {
        var newData = [Float](repeating: 0, count: data.count)
        for i in 0..<rows {
            for j in 0..<cols {
                newData[j * rows + i] = self[i, j]
            }
        }
        return Matrix(rows: cols, cols: rows, data: newData)
    }
    
    /// Matrix multiplication
    static func multiply(_ a: Matrix, _ b: Matrix) -> Matrix {
        assert(a.cols == b.rows, "Incompatible matrix dimensions")
        var result = Matrix(rows: a.rows, cols: b.cols, data: [Float](repeating: 0, count: a.rows * b.cols))
        for i in 0..<a.rows {
            for j in 0..<b.cols {
                var sum: Float = 0
                for k in 0..<a.cols {
                    sum += a[i, k] * b[k, j]
                }
                result[i, j] = sum
            }
        }
        return result
    }
    
    /// Matrix inverse (4x4 only)
    func inverse4x4() -> Matrix {
        assert(rows == 4 && cols == 4, "Inverse only implemented for 4x4 matrices")
        var inv = [Float](repeating: 0, count: 16)
        
        inv[0] = data[5] * data[10] * data[15] - data[5] * data[11] * data[14] - data[9] * data[6] * data[15] + data[9] * data[7] * data[14] + data[13] * data[6] * data[11] - data[13] * data[7] * data[10]
        inv[4] = -data[4] * data[10] * data[15] + data[4] * data[11] * data[14] + data[8] * data[6] * data[15] - data[8] * data[7] * data[14] - data[12] * data[6] * data[11] + data[12] * data[7] * data[10]
        inv[8] = data[4] * data[9] * data[15] - data[4] * data[11] * data[13] - data[8] * data[5] * data[15] + data[8] * data[7] * data[13] + data[12] * data[5] * data[11] - data[12] * data[7] * data[9]
        inv[12] = -data[4] * data[9] * data[14] + data[4] * data[10] * data[13] + data[8] * data[5] * data[14] - data[8] * data[6] * data[13] - data[12] * data[5] * data[10] + data[12] * data[6] * data[9]
        
        inv[1] = -data[1] * data[10] * data[15] + data[1] * data[11] * data[14] + data[9] * data[2] * data[15] - data[9] * data[3] * data[14] - data[13] * data[2] * data[11] + data[13] * data[3] * data[10]
        inv[5] = data[0] * data[10] * data[15] - data[0] * data[11] * data[14] - data[8] * data[2] * data[15] + data[8] * data[3] * data[14] + data[12] * data[2] * data[11] - data[12] * data[3] * data[10]
        inv[9] = -data[0] * data[9] * data[15] + data[0] * data[11] * data[13] + data[8] * data[1] * data[15] - data[8] * data[3] * data[13] - data[12] * data[1] * data[11] + data[12] * data[3] * data[9]
        inv[13] = data[0] * data[9] * data[14] - data[0] * data[10] * data[13] - data[8] * data[1] * data[14] + data[8] * data[2] * data[13] + data[12] * data[1] * data[10] - data[12] * data[2] * data[9]
        
        inv[2] = data[1] * data[6] * data[15] - data[1] * data[7] * data[14] - data[5] * data[2] * data[15] + data[5] * data[3] * data[14] + data[13] * data[2] * data[7] - data[13] * data[3] * data[6]
        inv[6] = -data[0] * data[6] * data[15] + data[0] * data[7] * data[14] + data[4] * data[2] * data[15] - data[4] * data[3] * data[14] - data[12] * data[2] * data[7] + data[12] * data[3] * data[6]
        inv[10] = data[0] * data[5] * data[15] - data[0] * data[7] * data[13] - data[4] * data[1] * data[15] + data[4] * data[3] * data[13] + data[12] * data[1] * data[7] - data[12] * data[3] * data[5]
        inv[14] = -data[0] * data[5] * data[14] + data[0] * data[6] * data[13] + data[4] * data[1] * data[14] - data[4] * data[2] * data[13] - data[12] * data[1] * data[6] + data[12] * data[2] * data[5]
        
        inv[3] = -data[1] * data[6] * data[11] + data[1] * data[7] * data[10] + data[5] * data[2] * data[11] - data[5] * data[3] * data[10] - data[9] * data[2] * data[7] + data[9] * data[3] * data[6]
        inv[7] = data[0] * data[6] * data[11] - data[0] * data[7] * data[10] - data[4] * data[2] * data[11] + data[4] * data[3] * data[10] + data[8] * data[2] * data[7] - data[8] * data[3] * data[6]
        inv[11] = -data[0] * data[5] * data[11] + data[0] * data[7] * data[9] + data[4] * data[1] * data[11] - data[4] * data[3] * data[9] - data[8] * data[1] * data[7] + data[8] * data[3] * data[5]
        inv[15] = data[0] * data[5] * data[10] - data[0] * data[6] * data[9] - data[4] * data[1] * data[10] + data[4] * data[2] * data[9] + data[8] * data[1] * data[6] - data[8] * data[2] * data[5]
        
        let det = data[0] * inv[0] + data[1] * inv[4] + data[2] * inv[8] + data[3] * inv[12]
        assert(abs(det) > 1e-6, "Matrix is singular")
        
        let invDet = 1.0 / det
        for i in 0..<16 {
            inv[i] *= invDet
        }
        return Matrix(rows: 4, cols: 4, data: inv)
    }
}

// MARK: - Core Functions

/// Compute unprojection matrix to transform Gaussians to Euclidean space.
///
/// - Parameters:
///   - extrinsics: The 4x4 extrinsics matrix of the camera view
///   - intrinsics: The 4x4 intrinsics matrix of the camera view
///   - imageShape: The (width, height) of the input image
/// - Returns: A 4x4 matrix to transform Gaussians from NDC space to Euclidean space
func getUnprojectionMatrix(
    extrinsics: Matrix,
    intrinsics: Matrix,
    imageShape: (width: Int, height: Int)
) -> Matrix {
    let imageWidth = Float(imageShape.width)
    let imageHeight = Float(imageShape.height)
    
    // Matrix converts OpenCV pixel coordinates to NDC coordinates
    let ndcMatrix = Matrix(rows: 4, cols: 4, data: [
        2.0 / imageWidth, 0.0, -1.0, 0.0,
        0.0, 2.0 / imageHeight, -1.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ])
    
    let product = Matrix.multiply(ndcMatrix, Matrix.multiply(intrinsics, extrinsics))
    return product.inverse4x4()
}

/// Unproject Gaussians from NDC space to world coordinates.
func unprojectGaussians(
    gaussiansNdc: Gaussians3D,
    extrinsics: Matrix,
    intrinsics: Matrix,
    imageShape: (width: Int, height: Int)
) -> Gaussians3D {
    let unprojectionMatrix = getUnprojectionMatrix(extrinsics: extrinsics, intrinsics: intrinsics, imageShape: imageShape)
    // Extract 3x4 affine transform
    let transform = Matrix(rows: 3, cols: 4, data: Array(unprojectionMatrix.data[0..<12]))
    return applyTransform(gaussians: gaussiansNdc, transform: transform)
}

/// Apply an affine transformation to 3D Gaussians.
///
/// - Note: This operation is not differentiable.
func applyTransform(gaussians: Gaussians3D, transform: Matrix) -> Gaussians3D {
    var result = gaussians
    
    // Extract linear and offset parts
    let transformLinear = Matrix(rows: 3, cols: 3, data: Array(transform.data[0..<9]))
    let transformOffset = [transform[0, 3], transform[1, 3], transform[2, 3]]
    
    // Transform mean vectors
    for i in 0..<gaussians.meanVectors.count {
        var newMean = [Float](repeating: 0, count: 3)
        for j in 0..<3 {
            newMean[j] = transformOffset[j]
            for k in 0..<3 {
                newMean[j] += gaussians.meanVectors[i][k] * transformLinear[k, j]
            }
        }
        result.meanVectors[i] = newMean
    }
    
    // Transform covariance matrices
    let covarianceMatrices = composeCovarianceMatrices(
        quaternions: gaussians.quaternions,
        singularValues: gaussians.singularValues
    )
    
    var transformedCovariances = covarianceMatrices
    for idx in 0..<covarianceMatrices.count {
        let cov = covarianceMatrices[idx]
        var transformed = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
        
        for i in 0..<3 {
            for j in 0..<3 {
                for k in 0..<3 {
                    for l in 0..<3 {
                        transformed[i][j] += transformLinear[i, k] * cov[k][l] * transformLinear[j, l]
                    }
                }
            }
        }
        transformedCovariances[idx] = transformed
    }
    
    let (quaternions, singularValues) = decomposeCovarianceMatrices(covarianceMatrices: transformedCovariances)
    result.quaternions = quaternions
    result.singularValues = singularValues
    
    return result
}

/// Decompose 3D covariance matrices into quaternions and singular values.
///
/// - Note: This operation is not differentiable.
func decomposeCovarianceMatrices(
    covarianceMatrices: [[[Float]]]
) -> (quaternions: [[Float]], singularValues: [[Float]]) {
    var quaternions: [[Float]] = []
    var singularValues: [[Float]] = []
    
    for cov in covarianceMatrices {
        // Simplified SVD using eigenvalue decomposition
        let (rotation, singularVals) = simpleSVD3x3(matrix: cov)
        quaternions.append(rotationMatrixToQuaternion(rotation))
        singularValues.append([sqrt(singularVals[0]), sqrt(singularVals[1]), sqrt(singularVals[2])])
    }
    
    return (quaternions, singularValues)
}

/// Compose 3D covariance matrices from quaternions and singular values.
func composeCovarianceMatrices(
    quaternions: [[Float]],
    singularValues: [[Float]]
) -> [[[Float]]] {
    var covariances: [[[Float]]] = []
    
    for i in 0..<quaternions.count {
        let rotation = quaternionToRotationMatrix(quaternions[i])
        let scales = singularValues[i]
        
        var diagonal = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
        diagonal[0][0] = scales[0] * scales[0]
        diagonal[1][1] = scales[1] * scales[1]
        diagonal[2][2] = scales[2] * scales[2]
        
        let cov = matrixMultiply3x3(rotation, matrixMultiply3x3(diagonal, transposeMatrix3x3(rotation)))
        covariances.append(cov)
    }
    
    return covariances
}

// MARK: - Quaternion & Rotation Operations

func rotationMatrixToQuaternion(_ m: [[Float]]) -> [Float] {
    let trace = m[0][0] + m[1][1] + m[2][2]
    var q = [Float](repeating: 0, count: 4)
    
    if trace > 0 {
        let s = 0.5 / sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (m[2][1] - m[1][2]) * s
        q[1] = (m[0][2] - m[2][0]) * s
        q[2] = (m[1][0] - m[0][1]) * s
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        q[3] = (m[2][1] - m[1][2]) / s
        q[0] = 0.25 * s
        q[1] = (m[0][1] + m[1][0]) / s
        q[2] = (m[0][2] + m[2][0]) / s
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        q[3] = (m[0][2] - m[2][0]) / s
        q[0] = (m[0][1] + m[1][0]) / s
        q[1] = 0.25 * s
        q[2] = (m[1][2] + m[2][1]) / s
    } else {
        let s = 2.0 * sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        q[3] = (m[1][0] - m[0][1]) / s
        q[0] = (m[0][2] + m[2][0]) / s
        q[1] = (m[1][2] + m[2][1]) / s
        q[2] = 0.25 * s
    }
    return q
}

func quaternionToRotationMatrix(_ q: [Float]) -> [[Float]] {
    let x = q[0], y = q[1], z = q[2], w = q[3]
    let xx = x * x, yy = y * y, zz = z * z
    let xy = x * y, xz = x * z, yz = y * z
    let wx = w * x, wy = w * y, wz = w * z
    
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ]
}

// MARK: - Matrix Utilities

func matrixMultiply3x3(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    var result = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
    for i in 0..<3 {
        for j in 0..<3 {
            for k in 0..<3 {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    return result
}

func transposeMatrix3x3(_ m: [[Float]]) -> [[Float]] {
    return [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]]
    ]
}

func simpleSVD3x3(matrix: [[Float]]) -> (rotation: [[Float]], singularValues: [Float]) {
    // Simplified SVD using power iteration
    // For production use, integrate Accelerate framework
    let m = matrix
    let mT = transposeMatrix3x3(m)
    let mTm = matrixMultiply3x3(mT, m)
    
    var eigenvalues = [Float](repeating: 0, count: 3)
    var eigenvectors = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 3)
    
    // Simplified eigenvalue extraction (approximation)
    eigenvalues[0] = mTm[0][0]
    eigenvalues[1] = mTm[1][1]
    eigenvalues[2] = mTm[2][2]
    
    eigenvectors[0] = [1, 0, 0]
    eigenvectors[1] = [0, 1, 0]
    eigenvectors[2] = [0, 0, 1]
    
    return (rotation: m, singularValues: eigenvalues)
}

// MARK: - Spherical Harmonics Conversion

/// Convert degree-0 spherical harmonics to RGB.
///
/// Reference: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
func convertSphericalHarmonicsToRGB(_ sh0: [[Float]]) -> [[Float]] {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return sh0.map { row in
        row.map { $0 * coeffDegree0 + 0.5 }
    }
}

/// Convert RGB to degree-0 spherical harmonics.
func convertRGBToSphericalHarmonics(_ rgb: [[Float]]) -> [[Float]] {
    let coeffDegree0 = sqrt(1.0 / (4.0 * Float.pi))
    return rgb.map { row in
        row.map { ($0 - 0.5) / coeffDegree0 }
    }
}

// MARK: - PLY File I/O

/// Loads a PLY file containing Gaussians.
func loadPLY(path: URL) -> (gaussians: Gaussians3D, metadata: SceneMetaData)? {
    guard let fileContent = try? String(contentsOf: path, encoding: .utf8) else {
        return nil
    }
    
    let lines = fileContent.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    var headerEnd = 0
    var vertexCount = 0
    
    // Parse header
    for (index, line) in lines.enumerated() {
        if line.starts(with: "element vertex") {
            vertexCount = Int(line.split(separator: " ").last ?? "0") ?? 0
        }
        if line == "end_header" {
            headerEnd = index + 1
            break
        }
    }
    
    guard vertexCount > 0 else { return nil }
    
    // Parse vertices (simplified binary PLY parsing)
    var meanVectors: [[Float]] = []
    var colors: [[Float]] = []
    var scales: [[Float]] = []
    var quaternions: [[Float]] = []
    var opacities: [Float] = []
    
    // Parse ASCII data
    for i in headerEnd..<min(headerEnd + vertexCount, lines.count) {
        let parts = lines[i].split(separator: " ").compactMap { Float($0) }
        if parts.count >= 11 {
            meanVectors.append([parts[0], parts[1], parts[2]])
            colors.append([parts[3], parts[4], parts[5]])
            opacities.append(parts[6])
            scales.append([parts[7], parts[8], parts[9]])
            if parts.count >= 14 {
                quaternions.append([parts[10], parts[11], parts[12], parts[13]])
            } else {
                quaternions.append([0, 0, 0, 1])
            }
        }
    }
    
    let rgbColors = convertSphericalHarmonicsToRGB(colors)
    
    let gaussians = Gaussians3D(
        meanVectors: meanVectors,
        singularValues: scales,
        quaternions: quaternions,
        colors: rgbColors,
        opacities: opacities
    )
    
    let metadata = SceneMetaData(
        focalLengthPx: 512.0,
        resolutionPx: (width: 640, height: 480),
        colorSpace: "linearRGB"
    )
    
    return (gaussians, metadata)
}

/// Save Gaussians to a PLY file.
func savePLY(
    gaussians: Gaussians3D,
    focalLength: Float,
    imageShape: (width: Int, height: Int),
    path: URL
) {
    var plyContent = "ply\n"
    plyContent += "format ascii 1.0\n"
    plyContent += "element vertex \(gaussians.meanVectors.count)\n"
    plyContent += "property float x\n"
    plyContent += "property float y\n"
    plyContent += "property float z\n"
    plyContent += "property float f_dc_0\n"
    plyContent += "property float f_dc_1\n"
    plyContent += "property float f_dc_2\n"
    plyContent += "property float opacity\n"
    plyContent += "property float scale_0\n"
    plyContent += "property float scale_1\n"
    plyContent += "property float scale_2\n"
    plyContent += "property float rot_0\n"
    plyContent += "property float rot_1\n"
    plyContent += "property float rot_2\n"
    plyContent += "property float rot_3\n"
    plyContent += "element image_size 2\n"
    plyContent += "property uint image_size\n"
    plyContent += "element intrinsic 9\n"
    plyContent += "property float intrinsic\n"
    plyContent += "element extrinsic 16\n"
    plyContent += "property float extrinsic\n"
    plyContent += "end_header\n"
    
    // Write vertex data
    let harmonicsColors = convertRGBToSphericalHarmonics(gaussians.colors)
    
    for i in 0..<gaussians.meanVectors.count {
        let mean = gaussians.meanVectors[i]
        let color = harmonicsColors[i]
        let scale = gaussians.singularValues[i]
        let quat = gaussians.quaternions[i]
        
        plyContent += "\(mean[0]) \(mean[1]) \(mean[2]) "
        plyContent += "\(color[0]) \(color[1]) \(color[2]) "
        plyContent += "\(gaussians.opacities[i]) "
        plyContent += "\(log(scale[0])) \(log(scale[1])) \(log(scale[2])) "
        plyContent += "\(quat[0]) \(quat[1]) \(quat[2]) \(quat[3])\n"
    }
    
    // Write metadata
    plyContent += "\(imageShape.width) \(imageShape.height)\n"
    plyContent += "\(focalLength) 0 \(Float(imageShape.width) * 0.5) 0 \(focalLength) \(Float(imageShape.height) * 0.5) 0 0 1\n"
    
    // Write identity extrinsic
    for i in 0..<16 {
        plyContent += (i % 4 == 0 && i != 0 ? "0 " : (i == 0 || i == 5 || i == 10 || i == 15 ? "1 " : "0 "))
    }
    plyContent += "\n"
    
    try? plyContent.write(to: path, atomically: true, encoding: .utf8)
}
