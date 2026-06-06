"""
Traveling Salesman Problem Environment
=====================================

Compatible with:
    dqn_pl_generic_v2.py

This environment implements a Reinforcement Learning formulation
of the Traveling Salesman Problem (TSP).

Cities:
    {(23, 45), (57, 12), (38, 78), (92, 34), (45, 67),
     (18, 90), (72, 55), (66, 24), (83, 62), (49, 40)}

Reward:

    normalized_reward = 100 * (1 - d_ij / d_max)

Where:
    d_ij  = distance between current city i and next city j
    d_max = maximum pairwise distance among all cities

The agent learns to:
    - choose next city (value action)
    - choose route position (position action)

The environment is fully compatible with the dual-head DQN:
    output1 -> city selection
    output2 -> position selection

============================================================
STATE REPRESENTATION
============================================================

State vector shape: (5,)

[state]
    [
        current_city_x_normalized,
        current_city_y_normalized,
        visited_ratio,
        total_distance_ratio,
        current_step_ratio,
    ]

============================================================
"""

import math
import numpy as np


# =========================================================
# BASE RL INTERFACE
# =========================================================

class RLProblem:

    @property
    def num_value_actions(self):
        raise NotImplementedError

    @property
    def num_position_actions(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, value_action, position_action):
        raise NotImplementedError


# =========================================================
# TSP ENVIRONMENT
# =========================================================

class TravelingSalesmanProblem(RLProblem):

    def __init__(self):

        # -------------------------------------------------
        # Cities
        # -------------------------------------------------

        self.cities = np.array(
            [
                (23, 45),
                (57, 12),
                (38, 78),
                (92, 34),
                (45, 67),
                (18, 90),
                (72, 55),
                (66, 24),
                (83, 62),
                (49, 40),
            ],
            dtype=np.float32,
        )

        self.num_cities = len(self.cities)

        # -------------------------------------------------
        # Distance matrix
        # -------------------------------------------------

        self.distance_matrix = self._build_distance_matrix()

        self.d_max = np.max(self.distance_matrix)

        self.max_possible_distance = (
            self.d_max * self.num_cities
        )

        self.reset()

    # =====================================================
    # ACTION SPACES
    # =====================================================

    @property
    def num_value_actions(self):
        """
        Choose next city.
        """
        return self.num_cities

    @property
    def num_position_actions(self):
        """
        Route insertion position.
        """
        return self.num_cities

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.visited = set()

        self.route = []

        self.total_distance = 0.0

        self.current_step = 0

        # Random initial city
        self.current_city = np.random.randint(
            0,
            self.num_cities,
        )

        self.route.append(self.current_city)

        self.visited.add(self.current_city)

        return self._get_state()

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        value_action,
        position_action,
    ):

        done = False

        next_city = int(value_action)

        # -------------------------------------------------
        # Invalid move
        # -------------------------------------------------

        if next_city in self.visited:

            reward = -100.0

            return (
                self._get_state(),
                reward,
                done,
                {
                    "invalid": True,
                },
            )

        # -------------------------------------------------
        # Distance
        # -------------------------------------------------

        d_ij = self.distance_matrix[
            self.current_city,
            next_city,
        ]

        # -------------------------------------------------
        # Reward
        # -------------------------------------------------

        reward = 100.0 * (
            1.0 - (d_ij / self.d_max)
        )

        # -------------------------------------------------
        # Update route
        # -------------------------------------------------

        insert_position = min(
            int(position_action),
            len(self.route),
        )

        self.route.insert(
            insert_position,
            next_city,
        )

        self.visited.add(next_city)

        self.total_distance += d_ij

        self.current_city = next_city

        self.current_step += 1

        # -------------------------------------------------
        # Finished route
        # -------------------------------------------------

        if len(self.visited) == self.num_cities:

            done = True

            # Return to initial city
            start_city = self.route[0]

            final_distance = self.distance_matrix[
                self.current_city,
                start_city,
            ]

            self.total_distance += final_distance

            final_reward = 100.0 * (
                1.0 - (final_distance / self.d_max)
            )

            reward += final_reward

            # Bonus for shorter tours
            efficiency_bonus = (
                1000.0
                * (
                    1.0
                    - (
                        self.total_distance
                        / self.max_possible_distance
                    )
                )
            )

            reward += efficiency_bonus

        return (
            self._get_state(),
            reward,
            done,
            {
                "route": self.route,
                "total_distance": self.total_distance,
            },
        )

    # =====================================================
    # STATE
    # =====================================================

    def _get_state(self):

        current_x, current_y = self.cities[
            self.current_city
        ]

        max_coord = np.max(self.cities)

        visited_ratio = (
            len(self.visited)
            / self.num_cities
        )

        distance_ratio = (
            self.total_distance
            / self.max_possible_distance
        )

        step_ratio = (
            self.current_step
            / self.num_cities
        )

        state = np.array(
            [
                current_x / max_coord,
                current_y / max_coord,
                visited_ratio,
                distance_ratio,
                step_ratio,
            ],
            dtype=np.float32,
        )

        return state

    # =====================================================
    # DISTANCE MATRIX
    # =====================================================

    def _build_distance_matrix(self):

        n = self.num_cities

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):

                if i == j:
                    continue

                matrix[i, j] = self._euclidean_distance(
                    self.cities[i],
                    self.cities[j],
                )

        return matrix

    # =====================================================
    # EUCLIDEAN DISTANCE
    # =====================================================

    @staticmethod
    def _euclidean_distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
        )

    # =====================================================
    # RENDER
    # =====================================================

    def render(self):

        print("--------------------------------")
        print("Current city:", self.current_city)
        print("Route:", self.route)
        print("Visited:", self.visited)
        print("Total distance:", self.total_distance)
        print("--------------------------------")


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    env = TravelingSalesmanProblem()

    state = env.reset()

    print("Initial state:")
    print(state)

    done = False

    while not done:

        value_action = np.random.randint(
            0,
            env.num_value_actions,
        )

        position_action = np.random.randint(
            0,
            env.num_position_actions,
        )

        next_state, reward, done, info = env.step(
            value_action,
            position_action,
        )

        print("Reward:", reward)

        env.render()

    print("Final route:")
    print(info["route"])

    print("Total distance:")
    print(info["total_distance"])


