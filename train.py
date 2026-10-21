import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 1. Cargar conjunto de datos de ejemplo
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2. Dividir en conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo de Regresión
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 4. Generar predicciones y calcular métricas
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 5. Guardar las métricas en un archivo de texto formateado en Markdown
with open("metrics.txt", "w") as f:
    f.write("## Métricas de Rendimiento del Modelo\n\n")
    f.write(f"- **MAE (Mean Absolute Error):** {mae:.4f}\n")
    f.write(f"- **MSE (Mean Squared Error):** {mse:.4f}\n")
    f.write(f"- **R² Score:** {r2:.4f}\n")

# 6. Generar y guardar el gráfico de Predicciones vs Valores Reales
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.3, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Valores Reales vs Predicciones")
plt.tight_layout()
plt.savefig("residuals.png") 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 1. Cargar conjunto de datos de ejemplo
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2. Dividir en conjuntos de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo de Regresión
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 4. Generar predicciones y calcular métricas
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# 5. Guardar las métricas en un archivo de texto formateado en Markdown
with open("metrics.txt", "w") as f:
    f.write("## Métricas de Rendimiento del Modelo\n\n")
    f.write(f"- **MAE (Mean Absolute Error):** {mae:.4f}\n")
    f.write(f"- **MSE (Mean Squared Error):** {mse:.4f}\n")
    f.write(f"- **R² Score:** {r2:.4f}\n")

# 6. Generar y guardar el gráfico de Predicciones vs Valores Reales
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.3, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Valores Reales vs Predicciones")
plt.tight_layout()
plt.savefig("residuals.png")