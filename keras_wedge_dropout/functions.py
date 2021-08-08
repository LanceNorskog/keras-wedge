import numpy as np

def wedge_direct_comparison(img1, img2):
    # This is exactly what a GlobalAveragePooling2D layer does
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)

    visible1 = img1 > mean1
    visible2 = img2 > mean2

    correlated = visible1 == visible2

    percentage = sum(correlated.flatten()) / len(img1.flatten())

    return correlated, percentage


def wedge_normalize(img1, img2):
    # normalize values in both images now range from 0 to 1 
    # the mean of all possible images is 0.5
    # somehow, there is no numpy function for this
    norm1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    norm2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

    # create a new matrix by Hadamard multiplication, elementwise
    # this exaggerates large values in both feature maps
    compare = norm1[:, :] * norm2[:, :]

    # Find the number of values greater than average
    correlated = sum(compare > 0.5)

    # Find the percentage of values greater than average
    percentage = sum(correlated.flatten()) / len(img1.flatten())

    return compare > 0.5, percentage

def wedge_standardize1d(img1, img2):
    # "standardize" values to mean of 0 and stdev of 1, or 2-norm (largest singular value)
    zero1 = img1 - np.min(img1)
    norm1 = zero1[:, :] / (np.linalg.norm(zero1, ord=2) + 0.0001)
    zero2 = img2 - np.min(img2)
    norm2 = zero2[:, :] / (np.linalg.norm(zero2, ord=2) + 0.0001)
    
    # create a new matrix by Hadamard multiplication, elementwise
    # this exaggerates large values in both feature maps
    compare = norm1[:, :] * norm2[:, :]

    # Find values greater than median average
    mask = compare > np.median(compare)
    
    # Find the percentage of values greater than average
    percentage = sum(mask.flatten()) / len(mask.flatten())

    return mask, percentage


def wedge_standardize2d(img1, img2):
    # "standardize" values to mean of 0 and stdev of 1, or 2-norm (largest singular value)
    zero1 = img1 - np.min(img1)
    norm1 = zero1[:, :] / np.linalg.norm(zero1, ord='fro')
    zero2 = img2 - np.min(img2)
    norm2 = zero2[:, :] / np.linalg.norm(zero2, ord='fro')
    
    # create a new matrix by Hadamard multiplication, elementwise
    # this exaggerates large values in both feature maps
    compare = norm1[:, :] * norm2[:, :]

    # Find values greater than average
    mask = compare > np.mean(compare)
    
    # Find the percentage of values greater than average
    percentage = sum(mask.flatten()) / len(img1.flatten())

    return mask, percentage

if __name__ == '__main__':
    img1 = np.asarray([[1,2],[3,4]])
    img2 = np.asarray([[5,6],[7,3]])

    print('Feature Map #1')
    print(img1)
    print('Feature Map #2')
    print(img2)
    print()

    correlation1, percentage1 = wedge_direct_comparison(img1, img2)
    print('Correlation (count > mean): ')
    print(correlation1)
    print('  similarity score:', percentage1)
    correlation2, percentage2 = wedge_normalize(img1, img2)
    print('Correlation (normalized and multiplied): ')
    print(correlation2)
    print('  similarity score:', percentage2)
    correlation3, percentage3 = wedge_standardize1d(img1, img2)
    print('Correlation (standardize to norm): ')
    print(correlation3)
    print('  similarity score:', percentage3)
    correlation4, percentage4 = wedge_standardize2d(img1, img2)
    print('Correlation (standardize to norm): ')
    print(correlation4)
    print('  similarity score:', percentage4)
