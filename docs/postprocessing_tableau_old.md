# Tableau Licensing, Hosting, and Workflow Summary

## 🔹 Tableau Licensing & Access Summary

| **License Type** | **Access to Dashboards** | **Interaction Level** | **Can Connect to Local Data?** | **Can Edit/Create Dashboards?** | **Typical Use Case** |
|------------------|--------------------------|------------------------|-------------------------------|-------------------------------|-----------------------|
| **Viewer**       | ✅ Online (Tableau Server/Cloud) | Basic (filter, drill, export) | ❌ No                         | ❌ No                         | View-only users       |
| **Explorer**     | ✅ Online (Tableau Server/Cloud) | Advanced (edit, save views)  | ❌ No                         | ✅ From **published data** only   | Power users, analysts |
| **Creator**      | ✅ Online & Desktop        | Full access              | ✅ Yes                        | ✅ Full creation/edit rights | Developers, engineers |
| **Public Viewer**| ✅ Public internet (Tableau Public) | Basic (anyone can see it) | ❌ No                         | ❌ No                         | General public        |

**Published data** refers to a shared, governed data source that a Creator uploads to Tableau Cloud or Server. Other users can build dashboards using this shared data, but cannot modify the original data source itself.

---

## 🔹 Dashboard Hosting Options

| **Platform**       | **Where Data Is Stored** | **Private or Public?** | **Requires Login?** | **Supports Extracts?** |
|--------------------|--------------------------|------------------------|---------------------|------------------------|
| **Tableau Desktop**| Local (your computer)    | Private (offline)      | ❌ No                | ✅ Yes                  |
| **Tableau Server** | Company infrastructure   | ✅ Private              | ✅ Yes               | ✅ Yes                  |
| **Tableau Cloud**  | Tableau-hosted servers   | ✅ Private              | ✅ Yes               | ✅ Yes                  |
| **Tableau Public** | Tableau’s public servers | ❌ Public (for all)     | ❌ No                | ✅ Yes                  |

> 📝 **Note**: Tableau Server and Tableau Cloud offer the same features. The only difference is **who hosts the environment** — your company (Server) or Tableau (Cloud).


---

## 🔹 Data 

When a Creator publishes a data source to Tableau Server or Tableau Cloud, they’re choosing what data others will use and how that  data is accessed (live connection or extract). In our case, we will only be using **extracs**.

### **Extract**
- Tableau creates a `.hyper` file with a snapshot of the data.
- Stored locally or within Tableau Cloud/Server.
- Improves performance, works offline, but requires refresh for updates.

If the team is using a published data source, they can build dashboards using that data — but cannot change the data connection type unless they have a Creator license.

---

## 🔹 Tableau Public – Key Considerations

**Tableau Public** is a free version of Tableau that allows users to publish and share visualizations publicly.

- Public dashboards and data are visible to everyone.
- No login or data protection options.
- Not suitable for internal/company use.
- Ideal for portfolios, public data storytelling, community sharing.

---

## 🔹 How This Works for Our Team (Example)

1. We **create a shared dashboard** using a **Creator** license.
2. Team members can **download the workbook**, which includes an **extract** of the original data.
3. They can **open the workbook in Tableau Desktop**:
   - If they **only view or make simple edits** (e.g., change titles, filters) **without changing the data**, they could use a **Viewer or Explorer** license **on Tableau Server/Cloud** — but not Tableau Desktop.
   - If they want to **open and interact with the workbook locally**, they **need Tableau Desktop** — and only **Creator licenses** allow them to **replace or reconnect to their own local data sources** (e.g., update the .csv file inside the workbook).
4. Therefore, for team members to **modify the data source themselves** (e.g., swap in their own files), they would each need a **Creator** license.
"""
