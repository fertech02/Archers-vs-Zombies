
# ML Project February 2026: Knights Archers Zombies

This repository contains the project's code of the course "[Machine Learning: Project](https://onderwijsaanbod.kuleuven.be/syllabi/e/H0T25AE.htm)" (KU Leuven, Faculty of Engineering, Department of Computer Science, [DTAI Section](https://dtai.cs.kuleuven.be)). In the following code, we both train a cnn to perform zombie detection and a rl policy to maximize zombie kills.


## Local installation

- It is recommended to use a newly-created virtual environment to avoid dependency conflicts.


- Install Pettingzoo with the additional requirements for the Butterfly environments

    ```
    pip install 'pettingzoo[butterfly]'
    ```

- Install SuperSuit, which will help managing your environments:

    ```
    pip install supersuit
    ```

- Your agents will be dependent on some RL library. Here we provide an example for installing Ray RLlib:

    ```
    pip install 'ray[rllib]'
    ```

- All dependencies are also listed in the `requirements.txt` file (`pip install -r requirements.txt`). Or in the `pyproject.toml` file if you want to use `uv`.

You can (visually) test your installation by running:

```
python3 evaluation.py -l random_agent.py -s --distortion=5
```

### Paths

Make sure you **do not use relative paths** in your implementation to load your trained model, as this will fail when running your agent from a different directory. Best practice is to retrieve the absolute path to the module directory:

```python
package_directory = os.path.dirname(os.path.abspath(__file__))
```

Afterwards, you can load your resources based on this `package_directory`:

```python
model_file = os.path.join(package_directory, 'models', 'mymodel.pckl')
```

### Getting error during the installation of the requirements

Consider downgrading your python version to 3.12 or lower as some packages might not yet support the latest python version.

