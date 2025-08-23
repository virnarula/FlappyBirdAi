"""
This module contains training for the DQN value model and saving it to file.
Run headless by default to speed up iterations.
"""

import argparse
import os
import numpy as np
import torch

import game
import model as model_mod


def train(argv=None):
	parser = argparse.ArgumentParser(description="Train DQN Value model for Flappy Bird")
	parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
	parser.add_argument('--save_every', type=int, default=25, help='Save model every N episodes')
	parser.add_argument('--out_dir', type=str, default='.', help='Directory to save models')
	parser.add_argument('--headless', action='store_true', help='Run pygame headless (SDL dummy driver)')
	parser.add_argument('--debug', action='store_true', help='Enable concise debug logs while training')
	parser.add_argument('--debug_state_interval', type=int, default=0, help='If >0, log one STATE line every N frames')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
	parser.add_argument('--epsilon_decay_steps', type=int, default=50000, help='Steps for epsilon decay schedule')
	parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon for exploration')
	parser.add_argument('--min_buffer_size', type=int, default=2000, help='Minimum replay size before learning')
	parser.add_argument('--easy_mode', action='store_true', help='Widen gap and reduce gravity to ease early training')
	args = parser.parse_args(argv)

	# Seeds
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	# Prepare model
	val_model = model_mod.ValueLearning(state_size=6, action_size=2, epsilon_decay_steps=args.epsilon_decay_steps, epsilon_start=args.epsilon_start, min_buffer_size=args.min_buffer_size)
	val_model.set_training(True)

	best_score = -1
	os.makedirs(args.out_dir, exist_ok=True)

	for episode in range(1, args.episodes + 1):
		Game = game.game()
		score = Game.run_game(val_model, debug=args.debug, debug_file=None, debug_state_interval=args.debug_state_interval, train=True, headless=args.headless, easy_mode=args.easy_mode)

		best_score = max(best_score, score)
		print(f"Episode {episode} | Score: {score} | Best: {best_score}")

		# Periodic checkpoint
		if episode % args.save_every == 0:
			ckpt_path = os.path.join(args.out_dir, f"val_dqn_ep{episode}.pth")
			val_model.save(ckpt_path)
			print(f"Saved checkpoint: {ckpt_path}")

	# Final save
	final_path = os.path.join(args.out_dir, "val_dqn_final.pth")
	val_model.save(final_path)
	print(f"Saved final: {final_path}")


if __name__ == "__main__":
	train()