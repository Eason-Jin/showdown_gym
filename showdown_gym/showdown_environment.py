import os
import time
from typing import Any, Dict, List

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from showdown_gym.base_environment import BaseShowdownEnv
from poke_env.battle import PokemonType
from poke_env.battle.move import Move
from poke_env.battle.weather import Weather
from poke_env.battle.move_category import MoveCategory
from poke_env.player.player import Player
from poke_env.battle.side_condition import SideCondition


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return 5 + 4 + 4  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        if 0 <= action <= 4:
            # Switches [0, 4] map to [0, 4]
            return action
        elif 5 <= action <= 8:
            # Moves [5, 8] map to [6, 9]
            return action + 1
        elif 9 <= action <= 12:
            # Tera evolve moves [9, 12] map to [22, 25]
            return action + 13
        else:
            # Fail safe
            return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if battle.finished:
            if battle.won == "me":
                return 100.0
            else:
                return -100.0

        score = 0.0
        # HP
        score += sum(mon.current_hp_fraction for mon in battle.team.values())
        score -= sum(mon.current_hp_fraction for mon in battle.opponent_team.values())
        # Status
        score += 0.1 * \
            sum(1 for mon in battle.team.values() if mon.status is not None and not mon.fainted)
        score -= 0.1 * \
            sum(1 for mon in battle.opponent_team.values() if mon.status is not None and not mon.fainted)
        # Boosts
        score += 0.05 * sum(sum(boost for boost in mon.boosts.values()
                            if boost > 0) for mon in battle.team.values())
        score -= 0.05 * sum(sum(-boost for boost in mon.boosts.values() if boost < 0)
                            for mon in battle.opponent_team.values())
        # Type advantage
        if active and opponent:
            score += self._combat_effectiveness(
                active, opponent)
        # Hazards
        score += 0.1 * len(battle.opponent_side_conditions)
        score -= 0.1 * len(battle.side_conditions)
        # Remaining Pokémon
        score += 0.5 * sum(not mon.fainted for mon in battle.team.values())
        score -= 0.5 * \
            sum(not mon.fainted for mon in battle.opponent_team.values())

        # Super effective move bonus
        if active and opponent :
            best_effectiveness = 1.0
            for move in battle.available_moves:
                eff = opponent.damage_multiplier(move)
                if eff > best_effectiveness:
                    best_effectiveness = eff
            # Reward for having a super effective move available
            if best_effectiveness > 1.0:
                score += 0.3 * (best_effectiveness - 1.0)

        return score

    def _combat_effectiveness(self, active: Pokemon, opponent: Pokemon):
        score = 0
        score += max([opponent.damage_multiplier(t)
                      for t in active.types if t is not None])
        score -= max([active.damage_multiplier(t)
                      for t in opponent.types if t is not None])

        if active.base_stats["spe"] > opponent.base_stats["spe"]:
            score += 0.1
        elif active.base_stats["spe"] < opponent.base_stats["spe"]:
            score -= 0.1

        score += active.current_hp_fraction * 0.4
        score -= opponent.current_hp_fraction * 0.4

        return score

    def _encode_status(self, status: Status) -> int:
        return status.value if status is not None else 0

    def _encode_weather(self, weather: Dict[Weather, int], types: List[PokemonType]) -> int:
        if not weather:
            return 0
        weather_type = max(weather, key=weather.get)
        duration = weather[weather_type]
        base_code = weather_type.value * 10 + duration
        # If weather boosts a type the pokemon has, return a higher value
        if (weather_type == Weather.RAINDANCE and PokemonType.WATER in types) \
                or (weather_type == Weather.SUNNYDAY and PokemonType.FIRE in types) \
                or (weather_type == Weather.SANDSTORM and PokemonType.ROCK in types) \
                or (weather_type == Weather.HAIL and PokemonType.ICE in types):
            return base_code + 100
        return base_code

    def _encode_side_conditions(self, side_conditions: Dict[SideCondition, int], types: List[PokemonType]) -> int:
        if not side_conditions:
            return 0
        side_condition_type = max(side_conditions, key=side_conditions.get)
        duration = side_conditions[side_condition_type]
        base_code = side_condition_type.value * 10 + duration
        # If side condition affects the pokemon negatively, return a lower value
        if (side_condition_type == SideCondition.STEALTH_ROCK and
                any(t in [PokemonType.FIRE, PokemonType.ICE, PokemonType.FLYING, PokemonType.BUG] for t in types)) \
            or (side_condition_type == SideCondition.TOXIC_SPIKES and
                PokemonType.POISON not in types and PokemonType.STEEL not in types):
            return base_code - 100
        return base_code

    def _calc_stat(self, battle: AbstractBattle, mon: Pokemon, stat: str):
        boost = 1.0
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        base = ((2 * mon.base_stats[stat] + 31) + 5) * boost

        # Weather-based stat boosts
        if battle.weather:
            # Sandstorm: Rock-type Sp. Def boost
            if stat == "spd" and Weather.SANDSTORM in battle.weather and battle.weather[Weather.SANDSTORM] > 0:
                if PokemonType.ROCK in mon.types:
                    base *= 1.5
            # Snow: Ice-type Def boost
            if stat == "def" and (
                (Weather.SNOW in battle.weather and battle.weather[Weather.SNOW] > 0)
                # If you want to support Hail as well
                or (Weather.HAIL in battle.weather and battle.weather[Weather.HAIL] > 0)
            ):
                if PokemonType.ICE in mon.types:
                    base *= 1.5

        return base

    def _estimate_damage(self, battle: AbstractBattle, move: Move, attacker: Pokemon, defender: Pokemon, max_accuracy: bool = False) -> float:
        physical_ratio = self._calc_stat(battle, attacker, "atk") / \
            self._calc_stat(battle, defender, "def")
        special_ratio = self._calc_stat(battle, attacker, "spa") / \
            self._calc_stat(battle, defender, "spd")

        if max_accuracy:
            accuracy = 1.0
        else:
            accuracy = move.accuracy

        base_power = move.base_power

        if battle.weather:
            if Weather.RAINDANCE in battle.weather and battle.weather[Weather.RAINDANCE] > 0:
                if move.type == PokemonType.WATER:
                    base_power *= 1.5
                elif move.type == PokemonType.FIRE:
                    base_power *= 0.5
            elif Weather.SUNNYDAY in battle.weather and battle.weather[Weather.SUNNYDAY] > 0:
                if move.type == PokemonType.FIRE:
                    base_power *= 1.5
                elif move.type == PokemonType.WATER:
                    base_power *= 0.5

        estimated_damage = base_power * (1.5 if move.type in attacker.types else 1) * (physical_ratio if move.category ==
                                                                                       MoveCategory.PHYSICAL else special_ratio) * accuracy * move.expected_hits * defender.damage_multiplier(move)
        return estimated_damage
    
    def _estimate_hazard_damage(self, mon: Pokemon, side_conditions: Dict[SideCondition, int]) -> float:
        """
        Estimate the fraction of HP lost by this Pokémon if it is switched in, due to hazards.
        """
        damage = 0.0
        # Stealth Rock
        if SideCondition.STEALTH_ROCK in side_conditions:
            # Stealth Rock damage is 1/8 * type effectiveness to Rock
            rock_multiplier = mon.damage_multiplier(PokemonType.ROCK)
            damage += 0.125 * rock_multiplier
        # Spikes (up to 3 layers)
        if SideCondition.SPIKES in side_conditions and PokemonType.FLYING not in mon.types and mon.item != "Air Balloon":
            layers = min(3, side_conditions[SideCondition.SPIKES])
            if layers == 1:
                damage += 0.125
            elif layers == 2:
                damage += 0.1667
            elif layers == 3:
                damage += 0.25
        # Toxic Spikes (if not airborne or Steel/Poison type)
        # Not direct damage, so not included here
        # Sticky Web, etc. are not direct damage
        return damage

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 26

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        active: Pokemon = battle.active_pokemon
        opponent: Pokemon = battle.opponent_active_pokemon
        combat_effectiveness = self._combat_effectiveness(active, opponent)
        my_hp_frac = active.current_hp_fraction
        my_status = self._encode_status(active.status)
        opp_hp_frac = opponent.current_hp_fraction
        opp_status = self._encode_status(opponent.status)

        move_damages = [self._estimate_damage(
            battle, m, active, opponent, max_accuracy=False) for m in battle.available_moves]
        while len(move_damages) < 4:
            move_damages.append(0.0)

        switches_info = []
        for mon in battle.available_switches:
            type_advantage = self._combat_effectiveness(mon, opponent)
            health_frac = mon.current_hp_fraction
            # Calculate hazard damage for this mon if switched in
            hazard_damage = self._estimate_hazard_damage(mon, battle.side_conditions)
            switches_info.extend([type_advantage, health_frac, hazard_damage])
        while len(switches_info) < 15:
            switches_info.extend([0.0, 0.0, 0.0])

        weather = self._encode_weather(battle.weather, active.types)

        can_tera = 1.0 if battle.can_tera else 0.0

        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                [combat_effectiveness],  # 1 component for combat effectiveness
                [my_hp_frac],  # 1 component for the health fraction of the active pokemon
                [my_status],  # 1 component for the status of the active pokemon
                [opp_hp_frac],  # 1 component for the health fraction of the opponent active pokemon
                [opp_status],  # 1 component for the status of the opponent active pokemon
                move_damages,  # 4 components for the expected damage of each move
                switches_info,  # 15 components for the switches info (type_adv, hp, hazard) for up to 5 switches
                [weather],  # 1 component for the weather
                [can_tera],  # 1 component for whether can tera
            ]
        )

        return final_vector


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(
                account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(
                account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
