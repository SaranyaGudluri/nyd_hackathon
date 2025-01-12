document.getElementById('queryForm').addEventListener('submit', async (event) => {
    event.preventDefault();

    const query = document.getElementById('query').value;
    const source = document.getElementById('source').value;

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, source })
        });

        if (!response.ok) {
            throw new Error('Failed to fetch results');
        }

        const result = await response.json();
        const resultsDiv = document.getElementById('results');

        resultsDiv.innerHTML = '';
        if (result.error) {
            resultsDiv.innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
            return;
        }

        const topResult = result.results?.[0] || result.relevant_verses?.[0];
        if (topResult) {
            const { sanskrit, translation, chapter = 'N/A', verse = 'N/A', speaker = 'N/A' } = topResult;

            resultsDiv.innerHTML = `
                <h2>Top Result</h2>
                <div class="result-item">
                    <p><strong>Sanskrit:</strong> ${sanskrit}</p>
                    <p><strong>Translation:</strong> ${translation}</p>
                    ${chapter !== 'N/A' ? `<p><strong>Chapter:</strong> ${chapter}</p>` : ''}
                    ${verse !== 'N/A' ? `<p><strong>Verse:</strong> ${verse}</p>` : ''}
                    ${speaker !== 'N/A' ? `<p><strong>Speaker:</strong> ${speaker}</p>` : ''}
                </div>
            `;
        } else {
            resultsDiv.innerHTML = '<p>No results found.</p>';
        }

    } catch (error) {
        console.error('Error:', error);
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});
