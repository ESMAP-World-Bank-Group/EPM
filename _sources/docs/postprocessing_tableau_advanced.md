# Tableau Advanced Tutorial: How to modify Tableau visualization

As explained above, Tableau Desktop is available from the shared team VDI. This tutorial covers the basics to help you create your first dashboard. For an in-depth analysis, you can also watch this [video](https://www.youtube.com/watch?v=j8FSP8XuFyk).

A tutorial by Mehdi Mikou on how to modify specifically the EPM visualization dashboard is available on the team's Drive [here](https://worldbankgroup.sharepoint.com/:v:/r/teams/PowerSystemPlanning-WBGroup/Shared%20Documents/EPM/4.%20Developments/Tableau/Tutorial%20Tableau%20June%202025.mov?csf=1&web=1&e=wzOYrj).

## 1. Connect to a Data Source

1. On the **Start Page**, under **Connect**, choose your data source:
    
    - Excel, Text File, CSV
        
    - Database (e.g., MySQL, PostgreSQL)
        
    - Tableau Server or cloud sources
        
2. Browse and select your file or enter credentials for databases.
    
3. Tableau will open the **Data Source** tab, displaying the data preview.

## 2. Prepare the Data

- Rename fields by double-clicking headers.
    
- Change data types by clicking the data type icon.
    
- Create calculated fields if needed (`Analysis > Create Calculated Field`).

## 3. Build Visualizations (Sheets)

1. Click on a **new worksheet** (`Sheet 1`).
    
2. Drag fields from the **Data pane** into:
    
    - **Rows** and **Columns** to build charts
        
    - **Marks** (Color, Size, Label, etc.) for additional detail
        
3. Examples:
    
    - Bar chart: Drag `Category` to Columns and `Sales` to Rows
        
    - Map: Drag `Country` to the view

## 4. Create a Dashboard

1. Click the **New Dashboard** icon (`Dashboard 1`).
    
2. Drag sheets from the **Sheets pane** onto the dashboard area.
    
3. Use objects (e.g., Text, Image, Web) from the left sidebar to enhance your dashboard.
    
4. Adjust layout and sizing to fit your needs.

## 5. Add Interactivity

- Use **Filters**:
    
    - Drag a field to Filters on a worksheet, then show it on the dashboard.
        
- Use **Actions** (`Dashboard > Actions`) to:
    
    - Filter other views
        
    - Highlight data
        
    - Link to other sheets or websites

## 6. Save and Share

- Save your workbook as `.twb` or `.twbx`.
    
- Export as PDF or Image (`File > Export`).
    
- Publish to Tableau Server, Tableau Public, or Tableau Cloud.
        
