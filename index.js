const { predict } = require("./predict");

async function main() {
  // User input data
  const userInput = {
    age: 40,
    bmi: 25.0,
    HbA1c_level: 6.5,
    blood_glucose_level: 140,
    smoking_history: "never",
  };

  // Make predictions using the predict function from predict.js
  const predictions = await predict(userInput);

  // Print the predictions
  console.log("Predictions:", predictions);
}

main();
