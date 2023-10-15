# Fonctionnement de la classe Regression
## Description
Cette classe a pour vocation de calculer une régression selon un modèle fourni avec des données ayant une incertitudes selon les deux axes.

## How To
**(1) Créer un modèle**

Créer une classe de la forme donnée en exemple ci-dessous. Elle représente la fonction selon laquelle on veut ajuster le dataset (fonction linéaire, affine, etc.)

```python
class Linear():
    def __init__(self):
        self.expression = "Model: a1.x+a2"
        self.init_conds = np.array( [1,0] )
    
    def __call__(self, params, x):
        a,b = params
        return a*x+b
```

- `self.expression` : Chaîne de caractères qui sera retournée avec les résultats. Noter les paramètres `a1`, `a2`, etc.
- `self.init_conds` : Vecteur contenant les valeurs “standards” des paramètres. L’algorithme se base là dessus. Des paramètres proches du résultats accélère le processus. *(Ici, par exemple, on cherche à ajuster notre dataset à partir d’une droite d’équation $y=x$, puis on fait varier `a` et `b` pour ajuster.)*
- `__call__` : Fonction définissant à proprement parler le modèle avec `params` le vecteur des paramètres à ajuster.
    - `return a*x+b`
    - `return a*x`
    - `return a*x**2+b*x+c`
    - etc.

**(2) Créer une instance de `Regression`**

On débute notre régression en créant une nouvelle instance de `Regression` en passant en argument notre Modèle 

```python
reg_lin = Regression(Linear)
```

`reg_lin` contiendra toutes les informations de notre régression.

**(3) Ajouter nos données**

Deux méthodes: soit on entre nos valeurs directement dans Python sous formes de vecteurs NumPy, soit on les importe d’un fichier CSV.

```python
x = np.array([..., ..., ...])
y = np.array([..., ..., ...])
x_err = np.array([..., ..., ...])
y_err = np.array([..., ..., ...])

**reg_lin.set_data(x, y, x_err, y_err)**
```

Dans le cas d’une incertitude constante pour $x$ et/ou $y$, on peut écrire uniquement la valeur.

```python
**reg_lin.set_data_from_csv('data.csv')**
```

<aside>

⚠️ Le fichier CSV doit être sous la forme

`x    , y    , x_err    , y_err`

Avec des séparateurs `,`

</aside>

<aside>
⚠️ Ne pas entrer d’incertitudes nulle: entrer une faible valeur (négligeable)

</aside>

**(4) Faire la régression**

```python
params, params_err = **reg_lin.solve()**
```

- `params` : Vecteur contenant les paramètres du modèle ajusté.
- `params_err` : Vecteur contenant les incertitudes associées.

**(5) Afficher les paramètres du modèle**

Affiche les valeurs des paramètres accompagnés de leur incertitude et le coefficient de corrélation.

```python
reg_lin.show_results()
```

```
*******************************************************
Model: a1.x+a2
	a1 = 4.421 ± 0.13
	a2 = -1.199 ± 0.718
R² = 98.269%
*******************************************************
```

**(6) Tracer la régression**

Trace nos données, la régression et l’intervalle de confiance sur un graphe en échelle linéaire.

```python
reg_lin.lin_plot()
```
**(7) Exporter la régression en CSV**

Exporte la régression sous la forme `x    , y_regr    , y_min_conf    , y_max_conf` dans un fichier `regression_output.csv`.

```python
reg_lin.to_csv()
```
