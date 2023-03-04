from sklearn.cluster import DBSCAN
import numpy as np
import os
import pandas as pd

def main():
    img_r = ["F:\AI\data/23_2_28/10x/1", "F:\AI\data/23_2_28/10x/-1"]
    csv_r = [i + '_csv' for i in img_r]
    radius = 5

    for csv_root in csv_r:
        new_path = csv_root + '_cluster'
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        for name in os.listdir(csv_root):
            csv_path = os.path.join(csv_root, name)
            new_csv = os.path.join(new_path, name)
            df = pd.read_csv(csv_path)

            df = df.loc[:, ['X', 'Y']]
            XY = df.values[:]
            db = DBSCAN(eps=radius, min_samples=3).fit(XY)

            labels = db.labels_
            cluster_ = set(labels)
            # 2 types in total:single point and cluster center point

            single_pointIndex = []
            cluster_index = [[] for i in cluster_]
            n = len(labels)
            for i in range(0, n):
                # i means the index of the raw point, label means the which cluster the point belongs to
                label = labels[i]
                if label == -1:
                    single_pointIndex.append(i)
                else:
                    cluster_index[label].append(i)
            # new_df = df1 = pd.DataFrame(data=, columns=['X', 'Y'])
            new_df = df.loc[single_pointIndex, ['X', 'Y']]

            for cl in cluster_index:
                new_df.append((df.loc[cluster_index[0], ['X', 'Y']]).mean(), ignore_index=True)
            new_df.to_csv(new_csv)




if __name__ == '__main__':
    main()
