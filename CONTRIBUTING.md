# Contributing to Our Discord Bot

Thank you for your interest in contributing to our Discord bot! We're excited to have you join our community and help us build a bot that serves our needs and maybe even goes beyond. Whether you're fixing bugs, adding new features, or improving documentation, every contribution is valuable.

## Before You Contribute

First, read these guidelines. Before you begin making changes, state your intent to do so in an Issue or on the Discord.

- **Check Existing Issues and Pull Requests**: Before starting work on a contribution, please check the repository's issues and pull requests to see if someone else is already working on something similar.

- **Follow the Project's Coding Standards**: Make sure to follow any coding standards and practices that are specific to the project. If the project includes a linter configuration, use it to check your code. (need to add standards)

- **Test Your Changes**: Ensure that your changes do not break existing functionality. Add tests if you're introducing new features or fix existing tests if necessary.

## How to Contribute

1. **Fork the Repository**: Start by forking the repository to your GitHub account. This creates a personal copy for you to work on.

2. **Clone Your Fork**: Clone your fork to your local machine to start making changes.
   
   ```bash
   git clone https://github.com/meri-nova/discord.git
   ```

3. **Create a New Branch**: Before you make any changes, switch to a new branch specific to the feature or fix you're working on.
   
   ```bash
   git checkout -b your-feature-name
   ```

4. **Make Your Changes**: Implement your feature, fix, or documentation updates. Make sure to keep your changes as focused as possible. If you're fixing a bug or adding a feature, consider adding or updating tests as appropriate.

5. **Commit Your Changes**: Once you're satisfied with your work, commit your changes with a clear and descriptive commit message.
   
   ```bash
   git commit -am "Add a concise and descriptive commit message"
   ```

6. **Push to Your Fork**: Push your changes to your fork on GitHub0.
   
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Submit a Pull Request**: Go to the original repository you forked on GitHub, and you'll see a prompt to submit a pull request from your new branch. Fill out the pull request form with a clear description of your changes.

## Setting up the bot (need better title)

to add by madhav :)

## Testing

We use GitHub Actions to run tests on each pull request. You can run these tests yourself as well. Before running unit tests, make sure you install the testing dependencies with `pip3 install -r requirements-test.txt`. Then run the tests by running pytest in the root directory of the repository.
>We can use requirement.txt but I think best practise is to have a seperate requirements-test.txt for testing.

## Linting

Please run lint on your pull requests to make accepting the requests easier. To do this, run `pylint src` in the root directory of the repository. Note that even if lint is passing, additional style changes to your submission may be made during merging.

## Seeking Help

If you have any questions or need help with setting up the project locally, don't hesitate to ask for help by creating an issue in the repository. We're more than happy to assist you.

Thank you for contributing to our project! Your support is what makes our community great.
