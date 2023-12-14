import numpy as np


def __loadfile(data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    labels = None
    if labels_file:
        path_to_labels = os.path.join(root, base_folder, labels_file)
        with open(path_to_labels, "rb") as f:
            labels = np.load(f)  # 0-based

    path_to_data = os.path.join(root, base_folder, data_file)
    print('\nPath to data: ', path_to_data, '\n')
    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything =np.load(f)
        images = np.reshape(everything, (-1, 3, 32, 32))
        images = np.transpose(images, (0, 1, 3, 2))

    return images, labels

