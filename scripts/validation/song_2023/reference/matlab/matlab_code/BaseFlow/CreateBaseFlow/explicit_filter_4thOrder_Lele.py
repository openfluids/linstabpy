import numpy as np


def fourth_order_filter(array: np.ndarray, dir: str) -> np.ndarray:

    filtered_array = np.zeros(np.shape(array))

    if dir == "x":
        nx = np.shape(array)[0]
        filtered_array[0, :] = (
            15.0 / 16.0 * array[0, :]
            + 1.0 / 4.0 * array[1, :]
            - 3.0 / 8.0 * array[2, :]
            + 1.0 / 4.0 * array[3, :]
            - 1.0 / 16.0 * array[4, :]
        )
        filtered_array[1, :] = (
            1.0 / 16.0 * array[0, :]
            + 3.0 / 4.0 * array[1, :]
            + 3.0 / 8.0 * array[2, :]
            - 1.0 / 4.0 * array[3, :]
            + 1.0 / 16.0 * array[4, :]
        )
        filtered_array[2, :] = (
            -1.0 / 16.0 * array[0, :]
            + 1.0 / 4.0 * array[1, :]
            + 5.0 / 8.0 * array[2, :]
            + 1.0 / 4.0 * array[3, :]
            - 1.0 / 16.0 * array[4, :]
        )

        filtered_array[-1, :] = (
            15.0 / 16.0 * array[-1, :]
            + 1.0 / 4.0 * array[-2, :]
            - 3.0 / 8.0 * array[-3, :]
            + 1.0 / 4.0 * array[-4, :]
            - 1.0 / 16.0 * array[-5, :]
        )
        filtered_array[-2, :] = (
            1.0 / 16.0 * array[-1, :]
            + 3.0 / 4.0 * array[-2, :]
            + 3.0 / 8.0 * array[-3, :]
            - 1.0 / 4.0 * array[-4, :]
            + 1.0 / 16.0 * array[-5, :]
        )
        filtered_array[-3, :] = (
            -1.0 / 16.0 * array[-1, :]
            + 1.0 / 4.0 * array[-2, :]
            + 5.0 / 8.0 * array[-3, :]
            + 1.0 / 4.0 * array[-4, :]
            - 1.0 / 16.0 * array[-5, :]
        )

        for i in range(3, nx - 3, 1):
            filtered_array[i, :] = (
                1.0 / 2.0 * array[i, :]
                + 9.0 / 32.0 * (array[i + 1, :] + array[i - 1, :])
                - 1.0 / 32.0 * (array[i + 3, :] + array[i - 3, :])
            )

        return filtered_array

    if dir == "y":
        ny = np.shape(array)[1]
        filtered_array[:, 0] = (
            15.0 / 16.0 * array[:, 0]
            + 1.0 / 4.0 * array[:, 1]
            - 3.0 / 8.0 * array[:, 2]
            + 1.0 / 4.0 * array[:, 3]
            - 1.0 / 16.0 * array[:, 4]
        )
        filtered_array[:, 1] = (
            1.0 / 16.0 * array[:, 0]
            + 3.0 / 4.0 * array[:, 1]
            + 3.0 / 8.0 * array[:, 2]
            - 1.0 / 4.0 * array[:, 3]
            + 1.0 / 16.0 * array[:, 4]
        )
        filtered_array[:, 2] = (
            -1.0 / 16.0 * array[:, 0]
            + 1.0 / 4.0 * array[:, 1]
            + 5.0 / 8.0 * array[:, 2]
            + 1.0 / 4.0 * array[:, 3]
            - 1.0 / 16.0 * array[:, 4]
        )

        filtered_array[:, -1] = (
            15.0 / 16.0 * array[:, -1]
            + 1.0 / 4.0 * array[:, -2]
            - 3.0 / 8.0 * array[:, -3]
            + 1.0 / 4.0 * array[:, -4]
            - 1.0 / 16.0 * array[:, -5]
        )
        filtered_array[:, -2] = (
            1.0 / 16.0 * array[:, -1]
            + 3.0 / 4.0 * array[:, -2]
            + 3.0 / 8.0 * array[:, -3]
            - 1.0 / 4.0 * array[:, -4]
            + 1.0 / 16.0 * array[:, -5]
        )
        filtered_array[:, -3] = (
            -1.0 / 16.0 * array[:, -1]
            + 1.0 / 4.0 * array[:, -2]
            + 5.0 / 8.0 * array[:, -3]
            + 1.0 / 4.0 * array[:, -4]
            - 1.0 / 16.0 * array[:, -5]
        )

        for i in range(3, ny - 3, 1):
            filtered_array[:, i] = (
                1.0 / 2.0 * array[:, i]
                + 9.0 / 32.0 * (array[:, i + 1] + array[:, i - 1])
                - 1.0 / 32.0 * (array[:, i + 3] + array[:, i - 3])
            )

        return filtered_array
