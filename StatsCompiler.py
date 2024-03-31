import pandas as pd
import numpy as np

class StatsComp:
    def __init__(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def extract(self):
        attributes = self.training_data.groupby('block_id').apply(lambda x: self.aggregate(x)).values
        target_labels = self.labels['pts'].values
        return attributes, target_labels

    def aggregate(self, player_data):
        latest_age = player_data.age.values[-1]
        average_games_played = np.mean(player_data.gp.values)
        average_points = np.mean(player_data.pts.values)
        points_last_year = player_data.pts.values[-1]
        points_year_before_last = player_data.pts.values[-2]
        average_net_rating = np.mean(player_data.net_rating.values)
        average_true_shooting = np.mean(player_data.ts_pct.values)
        average_usage_rate = np.mean(player_data.usg_pct.values)

        aggregated_data = np.array([
            [latest_age, average_games_played, average_points, points_last_year,
             points_year_before_last, average_net_rating, average_true_shooting,
             average_usage_rate]
        ])
        return pd.DataFrame(aggregated_data)
