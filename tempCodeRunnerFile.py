    model = models.Sequential([
        layers.Reshape((input_size, 1), input_shape=(input_size,)),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='sigmoid')
    ])