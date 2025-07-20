// commitlint.config.js
module.exports = {
	extends: ["@commitlint/config-conventional"],
	rules: {
		"type-enum": [
			2,
			"always",
			[
				"build",
				"chore",
				"ci",
				"docs",
				"feat",
				"fix",
				"perf",
				"refactor",
				"revert",
				"style",
				"test",
				"typo",
			],
		],
		"subject-full-stop": [2, "never", "."],
		"header-max-length": [2, "always", 72],
		"body-leading-blank": [2, "always"],
		"footer-leading-blank": [2, "always"],
		"scope-case": [2, "always", "lower-case"],

		// --- New common rules ---

		// Enforce specific values for scope (e.g., module names, feature areas)
		"scope-enum": [
			2,
			"always",
			[
				"blog-posts", // For changes to blog post content
				"components", // For changes to UI components
				"styles", // For CSS/SCSS changes
				"layout", // For changes to overall page layout
				"config", // For changes to configuration files (Astro, Tailwind, Biome)
				"deps", // For dependency updates
				"seo", // For SEO related changes
				"cli", // For changes related to command-line tools
				null, // Allows scope to be empty
			],
		],

		// Enforce sentence-case for the subject
		"subject-case": [
			2,
			"always",
			"sentence-case", // e.g., "Add new feature" instead of "add new feature"
		],

		// Enforce max line length for the body
		"body-max-line-length": [2, "always", 100],

		// Enforce max line length for the footer
		"footer-max-line-length": [2, "always", 100],
	},
};
