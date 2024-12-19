
# ESMAP Energy Planning Group Library

This documentation offers guidance on electricity system planning and provides pythons tools to quickly produce planning assessment at the regional or national level.

It is a collection of tools and models developed by the Energy Planning Group at the World Bank's Energy Sector Management Assistance Program (ESMAP). The tools are designed to support the planning and analysis of power systems, including the integration of renewable energy sources, the assessment of power system flexibility, and the evaluation of power system expansion plans.

The main tool is the Electricity Planning Model (EPM), a long-term, multi-year, multi-zone capacity expansion and dispatch model. The objective of the model is to minimize the sum of fixed (including annualized capital costs) and variable generation costs (discounted for time) for all zones and all years considered.

## Tools

The library now includes the following tools:

- **Electricity Planning Model (EPM)**: A long-term, multi-year, multi-zone capacity expansion and dispatch model. The objective of the model is to minimize the sum of fixed (including annualized capital costs) and variable generation costs (discounted for time) for all zones and all years considered.

- **Representative Days**: A tool to determine representative days for a given year. The tool is based on previously developed GAMS code for the Poncelet algorithm. The objective has been to automate the process and make it more user-friendly.