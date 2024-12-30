function generateRecipe() {
  const prompt = document.getElementById('prompt').value;
  const data = { prompt: prompt };

  document.getElementById('loader').style.display = 'block';

  fetch('/submit_ingredients', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById('loader').style.display = 'none';
      const recipes = data.map((recipe, index) => `
        <div class="dish">
          <h3>Dish ${index + 1}:</h3>
          <p>${recipe}</p>
        </div>
      `).join('');
      document.getElementById('recipe').innerHTML = recipes;
    })
    .catch(error => {
      document.getElementById('loader').style.display = 'none';
      console.error('Error:', error);
      document.getElementById('recipe').innerHTML = '<p>Error generating recipes. Please try again.</p>';
    });
}
