
.. _installation:


Installation
============

This section is under construction.


..
   User Installation Guide
   -----------------------

   This guide will walk you through the process of installing ``cycleopt`` via ``pip``. To isolate the installation and avoid conflicts with other Python packages, it is recommended to create a dedicated Conda virtual environment.

   1. Ensure conda is installed:

      - Check if conda is installed in your terminal:

      .. code-block:: bash

         conda list

      - If installed packages do not appear, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

   2. Open a terminal or command prompt and create a new virtual environment named ``barotropy_env``:

      .. code-block:: bash

         conda create --name barotropy_env python=3.11

   3. Activate the newly created virtual environment:

      .. code-block:: bash

         conda activate barotropy

   4. Install ``barotropy`` using ``pip`` within the activated virtual environment:

      .. code-block:: bash

         pip install barotropy

   5. Verify the installation by running the following command in your terminal:

      .. code-block:: bash

         python -c "import barotropy; barotropy.print_package_info()"

      If the installation was successful, you should see the banner and package information displayed in the console output.



   Congratulations! You have now successfully installed barotropy in its own Conda virtual environment using pip. You're ready to start using the package for your Python projects.




..
   Developer Installation Guide
   ------------------------------

   This installation guide is intended for developers who wish to contribute to or modify the source code. It assumes that the developer is using a Linux distribution or Windows with Git Bash terminal to have access to Git and Linux-like commands.

   1. **Fork the repository:**

      Navigate to the `project's GitHub page <https://github.com/turbo-sim/barotropy>` and click the "Fork" button in the upper right corner of the repository page to create a copy of the repository under your own GitHub account.


   2. **Clone the forked repository:**

      Open your terminal and run the following command, replacing `<your-username>` with your GitHub username:

      .. code-block:: bash

         git clone https://github.com/<your-username>/<repository-name>.git

      Navigate into the cloned repository:

      .. code-block:: bash

         cd <repository-name>

   3. **Create a dedicated Conda virtual environment for development**:

      Check that conda is installed:

      .. code-block:: bash

         conda list

      If conda is not installed, `install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.
      
      Create dedicated virtual environment for the package:

      .. code-block:: bash

         conda env create --file environment.yaml

   4. **Activate the newly created virtual environment**:

      .. code-block:: bash

         conda activate barotropy_env

      
   5. **Use Poetry to install the dependencies required for development**:

      .. code-block:: bash

         poetry install

      Poetry is a powerful dependency manager that offers separation of user and developer dependencies, ensuring that only the necessary packages are installed based on the user's intent. Additionally, it simplifies the process of adding, updating, and removing dependencies, making it easier to maintain the project's requirements.


