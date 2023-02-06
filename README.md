# Riptide
A deep learning library authored by Arnav Komaragiri

[Paper Written in EECS Student Magazine 2021-2022 edition](https://drive.google.com/file/d/1Mrsi3H2t7u8lHXqsHflU7LiM_96PHeuT/view) 

Note: This isn't PyTorch, I know the syntax looks a lot like it, but I just really like PyTorch syntax.

Example Code:
```python
import riptide
import tqdm

import numpy as np

class DenseModel(riptide.layers.Module):
    def __init__(self, in_features, hidden_features, out_features):
        self.linear_head = riptide.layers.Linear(in_features, hidden_features, initialization="lecun", drop_connect=True, p=0.5)
        self.linear_tail = riptide.layers.Linear(hidden_features, out_features, initialization="xavier")

        self.act = riptide.layers.ELU()
        self.sigmoid = riptide.layers.Sigmoid()

    def forward(self, x):
        x = self.linear_head(x)
        x = self.act(x)

        x = self.linear_tail(x)
        x = self.sigmoid(x)
        return x
        
data_path = "HW2_datafiles"
img_path, label_path = "MNISTnumImages5000_balanced.txt", "MNISTnumLabels5000_balanced.txt"

batch_size = 40
if not os.path.exists("train.pkl") or not os.path.exists("test.pkl"):
    dataset, mean, std = load_dataset(os.path.join(data_path, img_path), os.path.join(data_path, label_path), standardize=True)
    train, test = stratified_train_test_split(dataset)

    one_hot_train, one_hot_test = one_hot_encode(train, 10), one_hot_encode(test, 10)

    batch_train, batch_test = batch_dataset(one_hot_train, batch_size), batch_dataset(one_hot_test, batch_size)
    save_dataset(batch_train, "train.pkl")
    save_dataset(batch_test, "test.pkl")
else:
    print("Loading Cached Datasets...")
    batch_train, batch_test = read_dataset("train.pkl"), read_dataset("test.pkl")

model = DenseModel(784, 100, 10)
optimizer = riptide.optim.Adam(model.parameters(), lr=0.0005, minibatch_size=minibatch_size)
for i in range(epochs):
    # train set
    model.train()

    avg_loss, num_correct = 0, 0
    with tqdm(batch_train, unit="batch") as t_batch:
        for batch in t_batch:
            idx = np.arange(len(batch))
            np.random.shuffle(idx)
            idx = idx[:minibatch_size]

            model.zero_grad()
            for j, data in enumerate(batch[idx]):
                img, label = data[0], data[1]

                out = model(img)
                loss = loss_fn(label, out)
                avg_loss += float(loss.data)
                t_batch.set_postfix(loss=avg_loss / (len(batch_train) * (j + 1)))
                num_correct += int(np.argmax(out.data) == np.argmax(label.data))
                loss.backward()
            optimizer.step()
```

Example Conditional Model:
```python
import riptide
import tqdm

import numpy as np

class DenseModel(riptide.layers.Module):
    def __init__(self, in_features, hidden_features, out_features):
        self.linear_head = riptide.layers.Linear(in_features, hidden_features, initialization="lecun", drop_connect=True, p=0.5)
        self.linear_tail = riptide.layers.Linear(hidden_features, out_features, initialization="xavier")
        self.rand_tail = riptide.layers.Linear(out_features, out_features, initialization="xavier")

        self.act = riptide.layers.ELU()
        self.sigmoid = riptide.layers.Sigmoid()

    def forward(self, x):
        x = self.linear_head(x)
        x = self.act(x)

        x = self.linear_tail(x)
        x = self.sigmoid(x)
        if np.random.uniform() > 0.5:
            x = self.rand_tail(x)
        return x
        
data_path = "HW2_datafiles"
img_path, label_path = "MNISTnumImages5000_balanced.txt", "MNISTnumLabels5000_balanced.txt"

batch_size = 40
if not os.path.exists("train.pkl") or not os.path.exists("test.pkl"):
    dataset, mean, std = load_dataset(os.path.join(data_path, img_path), os.path.join(data_path, label_path), standardize=True)
    train, test = stratified_train_test_split(dataset)

    one_hot_train, one_hot_test = one_hot_encode(train, 10), one_hot_encode(test, 10)

    batch_train, batch_test = batch_dataset(one_hot_train, batch_size), batch_dataset(one_hot_test, batch_size)
    save_dataset(batch_train, "train.pkl")
    save_dataset(batch_test, "test.pkl")
else:
    print("Loading Cached Datasets...")
    batch_train, batch_test = read_dataset("train.pkl"), read_dataset("test.pkl")

model = DenseModel(784, 100, 10)
optimizer = riptide.optim.Adam(model.parameters(), lr=0.0005, minibatch_size=minibatch_size)
for i in range(epochs):
    # train set
    model.train()

    avg_loss, num_correct = 0, 0
    with tqdm(batch_train, unit="batch") as t_batch:
        for batch in t_batch:
            idx = np.arange(len(batch))
            np.random.shuffle(idx)
            idx = idx[:minibatch_size]

            model.zero_grad()
            for j, data in enumerate(batch[idx]):
                img, label = data[0], data[1]

                out = model(img)
                loss = loss_fn(label, out)
                avg_loss += float(loss.data)
                t_batch.set_postfix(loss=avg_loss / (len(batch_train) * (j + 1)))
                num_correct += int(np.argmax(out.data) == np.argmax(label.data))
                loss.backward()
            optimizer.step()
```
