# Contributing

We welcome contributions to both the codebase and the documentation. Below are a few guidelines to help you get started.

## Contributing to the code

If you would like to modify or improve the code, please follow these principles:

- Use your own branch for custom modifications that are specific to your case study or data. These include small changes that don’t need to be integrated across all models.
- Update the `main` branch for general improvements such as bug fixes, performance enhancements, or new features that would benefit all users.
- Make sure your changes are well documented and tested.

## Guidelines for submitting pull requests

If your contribution is significant (bug fix, improvement, new feature), you should:

1. Create a new branch from `main` (for example: `feature-my-update`).
2. Make your changes on that branch.
3. Open a pull request (PR) to merge your branch into `main`.

A pull request is a safe and documented way to propose changes. It allows you and others to:

- See exactly what was changed.
- Review and test the code before merging.
- Discuss the changes if needed.

Even if you are the only contributor, using pull requests helps keep a clean and transparent history, and makes it easier to undo changes if needed.

Pull requests should be used for important changes. For small edits, you can update the `main` branch directly, as long as it is up to date. Refer to previous sections for instructions on committing and pushing to `main`.

We recommend that pull requests:

- Include a short description of what was changed and why.
- Contain only one type of change (avoid mixing bug fixes with new features).
- Be tested before being submitted.

## Contributing to the documentation

All markdown files for the documentation are in the `docs/` folder. To contribute:

- You can improve an existing page by editing its `.md` file. For example, update `docs/model_formulation.md` to improve the model formulation section.
- You can add a new page by creating a new markdown file in `docs/`. If you do this, don’t forget to:
  1. Add the new file to the `_toc.yml` file so it appears in the documentation sidebar.

## How to update and improve the documentation

1. Go to the `docs/` folder.
2. Edit an existing markdown file or create a new one.
3. If you add a new file, update `_toc.yml` to include it in the navigation.
4. Add your changes to git and push them to the `main` branch.

Thank you for helping improve this project.
