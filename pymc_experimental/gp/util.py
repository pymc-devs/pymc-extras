import numpy as np


def build_XY(input_list, output_list=None, index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, index)])
    else:
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, range(num_outputs))])

    X = np.vstack(input_list)
    X = np.hstack([X, I[:, None]])

    return X, Y, I[:, None]  # slicesdef build_XY(input_list,output_list=None,index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, index)])
    else:
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, range(num_outputs))])

    X = np.vstack(input_list)
    X = np.hstack([X, I[:, None]])

    return X, Y, I[:, None]  # slices
