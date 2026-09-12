CREATE TABLE `episodes` (
	`id` text PRIMARY KEY NOT NULL,
	`series_id` text NOT NULL,
	`order` integer NOT NULL,
	`title` text NOT NULL,
	`video_url` text,
	`thumbnail_url` text,
	`coin_cost` integer DEFAULT 0 NOT NULL,
	`is_free` integer DEFAULT false NOT NULL,
	FOREIGN KEY (`series_id`) REFERENCES `series`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `purchases` (
	`id` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`play_order_id` text NOT NULL,
	`coins_granted` integer NOT NULL,
	`amount_paid` integer,
	`verified_at` integer,
	`status` text DEFAULT 'pending' NOT NULL,
	`product_id` text,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `purchases_play_order_id` ON `purchases` (`play_order_id`);--> statement-breakpoint
CREATE TABLE `series` (
	`id` text PRIMARY KEY NOT NULL,
	`title` text NOT NULL,
	`description` text DEFAULT '' NOT NULL,
	`cover_image_url` text DEFAULT '' NOT NULL,
	`is_published` integer DEFAULT false NOT NULL,
	`slug` text
);
--> statement-breakpoint
CREATE TABLE `subscriptions` (
	`user_id` text NOT NULL,
	`tier` text NOT NULL,
	`renews_at` integer,
	`status` text DEFAULT 'inactive' NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE TABLE `unlocked_episodes` (
	`user_id` text NOT NULL,
	`episode_id` text NOT NULL,
	`unlocked_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON UPDATE no action ON DELETE no action,
	FOREIGN KEY (`episode_id`) REFERENCES `episodes`(`id`) ON UPDATE no action ON DELETE no action
);
--> statement-breakpoint
CREATE UNIQUE INDEX `unlocked_episodes_pk` ON `unlocked_episodes` (`user_id`,`episode_id`);--> statement-breakpoint
CREATE TABLE `users` (
	`id` text PRIMARY KEY NOT NULL,
	`google_play_account_id` text,
	`device_id` text,
	`coin_balance` integer DEFAULT 0 NOT NULL,
	`created_at` integer NOT NULL
);
