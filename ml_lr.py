from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x=[[1],[2],[3],[4],[5]] #years of experience
y=[30000,35000,40000,50000,55000] #Salary

model = LinearRegression()
model.fit(x,y)

expected_sal=model.predict([[6.5]])
print(f"Expected Salary for 6.5 years is {expected_sal[0]}")

plt.scatter(x,y,color="red",label="Actaul Data")
plt.plot(x,model.predict(x),color="green",label="Regression Line")
plt.scatter(6.5,expected_sal,color="blue",label="Expected Salary")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.title("Salary Prediction")
plt.legend()
plt.grid(True)
plt.show()