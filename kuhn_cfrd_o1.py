from collections import defaultdict
import random
import numpy as np
import itertools

class KuhnPokerCFRD:
    def __init__(self):
        self.cards = ['J', 'Q', 'K']  # The deck of cards
        self.num_actions = 2  # Number of actions: Check (0) and Bet (1)
        self.actions = ['C', 'B']  # Actions
        self.regret_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.strategy_sum = defaultdict(lambda: np.zeros(self.num_actions))
        self.iterations = 100000  # Total number of iterations
        #self.checkpoints = [10, 100, 1000, 5000, 10000, 50000, 100000]  # Iterations to compute exploitability
        self.checkpoints = [10, 100, 1000] + [i*1000 for i in range(2,6)]

    def get_information_set_key(self, card, history):
        """
        Generates a unique key for each information set based on the player's card and the history of actions.
        """
        return card + ' ' + history

    def get_strategy(self, info_set_key):
        """
        Computes the current strategy from the regrets.
        """
        regrets = self.regret_sum.get(info_set_key, np.zeros(self.num_actions))
        positive_regrets = np.maximum(regrets, 1e-8)
        return positive_regrets / np.sum(positive_regrets)

    def get_average_strategy(self, info_set_key):
        """
        Computes the average strategy over all iterations at an information set.
        """
        strategy_sum = self.strategy_sum.get(info_set_key)
        if strategy_sum is None:
            return np.ones(self.num_actions) / self.num_actions
        return strategy_sum / np.sum(strategy_sum)

    def cfr_old(self, cards, history, p0, p1):
        """
        Performs CFR for the entire game tree.
        (p0, p1) are the reach probabilities.
        """
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        # Terminal states
        terminal_utility = self.get_payoff(history, cards)
        if terminal_utility is not None:
            return terminal_utility[player]

        info_set_key = self.get_information_set_key(cards[player], history)
        strategy = self.get_strategy(info_set_key)

        # Update strategy sum, weighted by reach probability
        self.strategy_sum[info_set_key] += (p0 if player == 0 else p1) * strategy

        util = np.zeros(self.num_actions)
        node_util = 0

        for i in range(self.num_actions):
            action = self.actions[i]
            next_history = history + action
            if player == 0:
                # This seems to be assuming zero sum, which it currently is,
                # but other parts of the code don't assume that.
                util[i] = -self.cfr(cards, next_history, p0 * strategy[i], p1)
            else:
                util[i] = -self.cfr(cards, next_history, p0, p1 * strategy[i])
            node_util += strategy[i] * util[i]

        # Update regrets, weighted by reach probability
        regrets = self.regret_sum.get(info_set_key, np.zeros(self.num_actions))
        for i in range(self.num_actions):
            regret = util[i] - node_util
            regrets[i] += (p1 if player == 0 else p0) * regret
        self.regret_sum[info_set_key] = regrets

        return node_util

    def cfr(self, cards, history, p0, p1):
        """Performs the CFR algorithm recursively without assuming zero-sum."""
        plays = len(history)
        player = plays % 2

        # Terminal state check
        terminal_utilities = self.get_payoff(history, cards)
        if terminal_utilities is not None:
            return terminal_utilities

        info_set = self.get_information_set_key(cards[player], history)
        strategy = self.get_strategy(info_set)
        reach_prob = p0 if player == 0 else p1
        self.strategy_sum[info_set] += reach_prob * strategy

        util = np.zeros((self.num_actions, 2))
        node_util = np.zeros(2)

        for i in range(self.num_actions):
            next_history = history + self.actions[i]
            if player == 0:
                next_p0, next_p1 = p0 * strategy[i], p1
            else:
                next_p0, next_p1 = p0, p1 * strategy[i]

            util[i] = self.cfr(cards, next_history, next_p0, next_p1)
            node_util += strategy[i] * util[i]

        # Regret update for the current player, weighted by opponent's reach probability
        opponent_prob = p1 if player == 0 else p0
        self.regret_sum[info_set] += opponent_prob * (util[:, player] - node_util[player])

        return node_util

    def br(self, card, history, player, op_range, indent=""):
        """
        Performs CFR for the entire game tree.
        """
        # Terminal states
        if self.is_terminal(history):
            value = 0
            for c, p in zip(self.cards, op_range):
                cards = (card, c) if player == 0 else (c, card)
                payoff = self.get_payoff(history, cards)
                value += p * payoff[player]
            opr = ", ".join([f"{c}: {p:.2f}" for c, p in zip(self.cards, op_range)])
            # print(f"{indent}Terminal: {history}, {card}, [{opr}] -> {value}")
            return value

        if len(history) % 2 == player:
            # Actually both players are maximizing since get_payoff returns the payoff for the player
            best_value, best_action = float('-inf'), None
            # print(f"{indent}Maximizing: {history}, {card} -> ...")
            for i in range(self.num_actions):
                value = self.br(card, history + self.actions[i], player, op_range, indent + "  ")
                if value >= best_value:
                    best_value, best_action = value, i
            # print(f"{indent}Best: {best_value:.4f} ({best_action})")
            return best_value

        value = 0
        # ps = []
        # for i in range(self.num_actions):
        #     s = []
        #     for c, p in zip(self.cards, op_range):
        #         strategy = self.get_average_strategy(self.get_information_set_key(c, history))
        #         s.append(strategy[i] * p)  # Pr[spiller i | holder c] * Pr[holder c]
        #     ps.append(sum(s))
        # rng = ", ".join([f"{c}: {p:.2f}" for c, p in zip(self.cards, op_range)])
        # ps = ", ".join([f"{a}: {p:.2f}" for a, p in zip(self.actions, ps)])
        # print(f"{indent}Averaging: {history}, [{rng}], [{ps}] -> ...")
        for i in range(self.num_actions):
            # Compute updated range, given opponent plays action i
            # Pr[holder c | spiller i] = Pr[spiller i | holder c] * Pr[holder c] / Pr[spiller i]
            new_op_range = []
            for c, p in zip(self.cards, op_range):
                # Get probability of opponent playing action i if having card c
                strategy = self.get_average_strategy(self.get_information_set_key(c, history))
                new_op_range.append(strategy[i] * p)  # Pr[spiller i | holder c] * Pr[holder c]
            prob = sum(new_op_range)  # Pr[spiller i] = sum_c(Pr[spiller i | holder c] * Pr[holder c])
            new_op_range = [x / prob for x in new_op_range]  # It's clear how reach probs may be easier
            a_value = self.br(card, history + self.actions[i], player, new_op_range, indent + "  ")
            value += prob * a_value
        # print(f"{indent}Average: {value:.4f}")
        return value

    def ev(self, history='', ranges=None):
        """
        ranges: (c1, c2) tensor
        """
        if ranges is None:
            ranges = (np.ones((3, 3)) - np.eye(3))/6

        # Terminal states
        if self.is_terminal(history):
            value = 0
            #for c1, c2 in itertools.combinations(self.cards, 2):
            for c1, c2 in itertools.product(range(len(self.cards)), repeat=2):
                p = ranges[c1, c2]
                payoff = self.get_payoff(history, (self.cards[c1], self.cards[c2]))
                value += p * np.array(payoff)  # (2,) values
            return value

        player = len(history) % 2
        value = 0
        for i in range(self.num_actions):
            # Compute updated range, given opponent plays action i
            # Pr[holder c | spiller i] = Pr[spiller i | holder c] * Pr[holder c] / Pr[spiller i]
            new_ranges = np.zeros((3, 3))
            for c1, c2 in itertools.product(range(len(self.cards)), repeat=2):
                c = self.cards[(c1, c2)[player]]
                # Get probability of playing action i if having card c
                s = self.get_average_strategy(self.get_information_set_key(c, history))[i]
                new_ranges[c1, c2] += s * ranges[c1, c2]
                # Basically, if player == 0 we are setting
                # new_ranges = ranges * strategy[i, :]
                # if player == 1 we are setting
                # new_ranges = ranges * strategy[:, i]

            # In principle we should normalize the ranges here. Otherwise they are reach
            # probabilities.
            # We don't normalize the new_ranges, as it would cancel with the final payoff calc anyway
            value += self.ev(history + self.actions[i], new_ranges)
        return value

    def best_response_value(self, player):
        """
        Computes the best response value for the specified player against the opponent's average strategy.
        """
        total_utility = 0
        for card1 in self.cards:
            op_range = [(1/2 if c != card1 else 0) for c in self.cards]
            total_utility += self.br(card1, '', player, op_range) / 3
        return total_utility

    def is_terminal(self, history):
        """
        Checks if a history corresponds to a terminal state.
        """
        # Terminal conditions in Kuhn poker
        if history in ['CC', 'BB', 'CBB', 'BC', 'CBC']:
            return True
        return False

    def get_payoff(self, history, cards):
        """Calculates the payoff at a terminal state."""
        if history in ['CC', 'BC', 'CBB', 'BB', 'CBC']:
            if history == 'CC':
                winner = self.winner(cards)
                return [1 if i == winner else -1 for i in range(2)]
            elif history == 'BC':
                return [1, -1]
            elif history in ['CBB', 'BB']:
                winner = self.winner(cards)
                return [2 if i == winner else -2 for i in range(2)]
            elif history == 'CBC':
                return [-1, 1]
        return None

    def winner(self, cards):
        """
        Determines the winner based on the cards.
        Returns the player index (0 or 1).
        """
        rank0 = self.cards.index(cards[0])
        rank1 = self.cards.index(cards[1])
        return 0 if rank0 > rank1 else 1


    def train(self):
        """
        Trains the CFR-D algorithm over the specified number of iterations.
        """
        for i in range(1, self.iterations + 1):
            cards = self.cards.copy()
            random.shuffle(cards)
            self.cfr(cards, '', 1, 1)
            if i in self.checkpoints:
                ev = self.ev()
                expl0 = self.best_response_value(player = 0) - ev[0]
                expl1 = self.best_response_value(player = 1) - ev[1]
                print(f"Iteration {i}: Objective: {ev[0]:.4f}, Exploitability = ({expl0:.4f} + {expl1:.4f})/2 = {(expl0+expl1)/2:.4f}")
        average_game_value = self.ev()
        print(f"\nFinal Objective: {average_game_value[0]:.4f}, {average_game_value[1]:.4f}\n")
        self.print_strategy()

    def print_strategy(self):
        """
        Prints the average strategy in the desired format.
        """
        print("Optimized M1 (Player 1's initial strategy):")
        for card in self.cards:
            info_set_key = self.get_information_set_key(card, '')
            avg_strategy = self.get_average_strategy(info_set_key)
            print(f"{card}")
            print(f"  CHECK: {avg_strategy[0]:.4f}")
            print(f"  BET: {avg_strategy[1]:.4f}")
        print("\nOptimized T (Player 2's strategy):")
        for card in self.cards:
            # Player 2's strategies after Player 1's actions
            for history in ['C', 'B']:
                info_set_key = self.get_information_set_key(card, history)
                avg_strategy = self.get_average_strategy(info_set_key)
                if info_set_key in self.strategy_sum:
                    action_label = "CHECK -> " if history == 'C' else "BET -> "
                    print(f"{card}")
                    print(f"  {action_label}CHECK: {avg_strategy[0]:.4f}")
                    print(f"  {action_label}BET: {avg_strategy[1]:.4f}")
        print("\nOptimized M2 (Player 1's response strategy):")
        for card in self.cards:
            # Player 1's strategies after 'C' followed by 'B' (history 'CB')
            history = 'CB'
            info_set_key = self.get_information_set_key(card, history)
            avg_strategy = self.get_average_strategy(info_set_key)
            if info_set_key in self.strategy_sum:
                print(f"{card}")
                print(f"  CHECK -> BET -> FOLD: {avg_strategy[0]:.4f}")
                print(f"  CHECK -> BET -> CALL: {avg_strategy[1]:.4f}")

if __name__ == "__main__":
    cfrd = KuhnPokerCFRD()
    cfrd.train()
