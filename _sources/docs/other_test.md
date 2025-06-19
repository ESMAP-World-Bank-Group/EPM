# Gathering Data

<!-- Put this in an HTML or Markdown file in your GitHub Pages repo -->
<h2>Team Dashboard</h2>
<div id="dashboard-table"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>
<script>
fetch('dwld/Template_Data_Source.csv')
  .then(response => response.text())
  .then(csvText => {
    const data = Papa.parse(csvText, { header: true });
    const table = document.createElement('table');
    table.border = '1';
    
    // Headers
    const headerRow = document.createElement('tr');
    Object.keys(data.data[0]).forEach(col => {
      const th = document.createElement('th');
      th.innerText = col;
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Data Rows
    data.data.forEach(row => {
      if (Object.values(row).every(cell => cell === '')) return; // skip empty rows
      const tr = document.createElement('tr');
      Object.values(row).forEach(cell => {
        const td = document.createElement('td');
        td.innerText = cell;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    
    document.getElementById('dashboard-table').appendChild(table);
  });
</script>