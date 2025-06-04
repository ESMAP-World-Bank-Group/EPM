## About

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15591290.svg)](https://doi.org/10.5281/zenodo.15591290)


Full documentation is available [here](https://esmap-world-bank-group.github.io/EPM/home.html).

EPM (Electricity Planning Model) is a least-cost power system planning tool. EPM has been developed by the Power Systems Planning Group, embedded in the Energy Sector Management Assistance Program (ESMAP) of the World Bank. As power system planning is one of the key activities performed by energy ministries and utilities around the world, the ambition of this tool is to actively inform the operational work of the World Bank's staff and clients and to be an evolving and versatile decision-making tool. 

EPM is formulated in the General Algebraic Modeling System (GAMS). All input data is provided via an Excel interface to the model and likewise, the results are collected in an Excel output file. Knowledge of the GAMS programming language is not a prerequisite for the basic runs.
In short, EPM minimizes the costs of expanding and operating a power system while meeting the system's technical, economic, and environmental requirements. EPM is mostly used as a long-term planning model, which means it optimizes the capacity expansion on a yearly basis based on system costs over multiple years, including fixed costs (annualized capital and fixed operation and maintenance [O&M]), variable costs (variable O&M) and fuel costs. Moreover, EPM addresses the dispatch of the generators, decides on the activities per geographical zone and the exchange between them. The model also co-optimizes reserves and generation allowing to allocate spinning reserves among generators. It also allows to test the impact of different policies, e.g., emissions limits, fuel, and import limits, spinning reserve requirements, transmissions caps, ramp limits, or carbon prices, on the power system evolution and costs.

The practical deployment of EPM consists of a 7-step process, which is illustrated below.

<img src="https://i.postimg.cc/jdZH17w9/Screenshot-2022-01-14-102001.png" alt="drawing" width="300" align="center"/>

---

## Works deploying EPM

- Suski, A., Remy, T., Chattopadhyay, D., Song, C. S., Jaques, I., Keskes, T., & Li, Y. (2021). Analyzing Electric Vehicle Load Impact on Power Systems: Modeling Analysis and a Case Study for Maldives. IEEE Access, 9, 125640-125657.
- World Bank. 2021. The Value of Trade and Regional Investments in The Pan-Arab Electricity Market : Integrating Power Systems and Building Economies. World Bank, Washington, DC. © World Bank. https://openknowledge.worldbank.org/handle/10986/36614
- Timilsina, Govinda; Deluque Curiel, Ilka; Chattopadhyay, Deb. 2021. How Much Does Latin America Gain from Enhanced Cross-Border Electricity Trade in the Short Run?. Policy Research Working Paper;No. 9692. World Bank, Washington, DC. © World Bank. https://openknowledge.worldbank.org/handle/10986/35729
- Huang, Z., Smolenova, I., Chattopadhyay, D., Govindarajalu, C., De Wit, J., Remy, T., & Curiel, I. D. (2021). ACT on RE+ FLEX: Accelerating Coal Transition Through Repurposing Coal Plants Into Renewable and Flexibility Centers. IEEE Access, 9, 84811-84827.
- Chattopadhyay, D., Chitkara, P., Curiel, I. D., & Draugelis, G. (2020). Cross-Border Interconnectors in South Asia: Market-Oriented Dispatch and Planning. IEEE Access, 8, 120361-120374.
- Remy, T., & Chattopadhyay, D. (2020). Promoting better economics, renewables and CO2 reduction through trade: A case study for the Eastern Africa Power Pool. Energy for Sustainable Development, 57, 81-97.
- Islam, M. E., Khan, M. M. Z., Chattopadhyay, D., & Väyrynen, J. (2021). Impact of COVID-19 on dispatch and capacity plan: A case study for Bangladesh. The Electricity Journal, 34(5), 106955.
- Islam, M. E., Khan, M. M. Z., Chattopadhyay, D., & Draugelis, G. (2020, August). Economic benefits of cross border power trading: A case study for Bangladesh. In 2020 IEEE Power & Energy Society General Meeting (PESGM) (pp. 1-5). IEEE.
- Spyrou, E., Hobbs, B. F., Bazilian, M. D., & Chattopadhyay, D. (2019). Planning power systems in fragile and conflict-affected states. Nature energy, 4(4), 300-310.
- World Bank Group. Poland Energy Transition: The Path to Sustainability in the Electricity and Heating Sector. World Bank, 2018.


## Citing EPM

Please cite EPM as:

Chattopadhyay, D., De Sisternes, F., Oguah, S. K. W., World Bank Electricity Planning Model (EPM) Mathematical Formulation, 2018, Energy Sector Management Assistance Program (ESMAP), International Bank for Reconstruction and Development, The World Bank, Washington DC

```bibtex
@article{
  author = {Chattopadhyay, Debabrata and De Sisternes, Fernando and Oguah, Samuel Kofi Wilson},
  title = {World Bank Electricity Planning Model (EPM) Mathematical Formulation},
  year = {2018},
  institution = {Energy Sector Management Assistance Program (ESMAP), International Bank for Reconstruction and Development, The World Bank},
  address = {Washington DC}
}
```

## License

Licensed under the Areative Commons Zero v1.0 Universal (the "License"); you
may not use this file except in compliance with the License. You may
obtain a copy of the License at <https://creativecommons.org/publicdomain/zero/1.0/>
