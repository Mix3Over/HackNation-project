import { parse } from "csv-parse/sync";
import fs from "fs";
import { cache_wskaznikow } from "@/db/schema";
import { db } from "../db";

// CSV parsing and seeding function
async function seedCacheWskaznikow(
  csvFilePath: string = "/home/tyfon/Programming/HackNation-project/ProjectCore/Data/index_branż.csv",
) {
  try {
    console.log("Reading CSV file...");
    const csvContent = fs.readFileSync(csvFilePath, "utf-8");

    console.log("Parsing CSV...");
    const records = parse(csvContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true,
      delimiter: ";",
      cast: (value, context) => {
        // Handle special cases for numeric fields
        if (value === "" || value === null || value === undefined) return null;
        if (
          context.header &&
          [
            "revenue",
            "net_profit",
            "investments",
            "firms",
            "profitable_firms",
            "profit_margin",
            "profitable_share",
            "bank_debt",
            "total_liab",
            "leverage_bank_to_rev",
            "leverage_tot_to_rev",
            "invest_intensity",
            "cagr_revenue",
            "cagr_net_profit",
            "cagr_investments",
            "cagr_firms",
            "profit_margin_std_3y",
            "size_index",
            "growth_index",
            "profit_index",
            "risk_index",
            "outlook_index",
            "current_index",
            "future_index",
            "main_index",
            "class1",
          ].includes(context.header)
        ) {
          const num = parseFloat(value);
          return isNaN(num) ? null : num;
        }
        return value;
      },
    });

    console.log(`Parsed ${records.length} records`);

    // Generate unique ID as pkd_section + pkd_2025 + year
    const preparedRecords: any = records.map((val: any) => ({
      ...val,
      year: val.year,
      date: val.year,
      pkd:
        (val?.pkd_2025?.startsWith?.("0")
          ? val.pkd_2025.substr(1)
          : val.pkd_2025) ||
        (val?.PKD?.startsWith?.("0") ? val.PKD.substr(1) : val.PKD),
      id: crypto.randomUUID(),
    }));

    console.log(preparedRecords[0]);
    console.log("Clearing existing data...");
    // await db.delete(cache_wskaznikow);

    console.log("Inserting data...");
    for (const record of preparedRecords) {
      await db.insert(cache_wskaznikow).values(record).execute();
    }

    // console.log(
    //   `✅ Seeded ${result.rows.length} records into cache_wskaznikow table [file:1]`,
    // );
  } catch (error) {
    console.error("❌ Seeding failed:", error);
    throw error;
  } finally {
    // await pool.end();
  }
}

// Run the seeder
if (require.main === module) {
  seedCacheWskaznikow()
    // seedCacheWskaznikow("/home/tyfon/Downloads/index_branż.csv")
    .then(() => {
      seedCacheWskaznikow("/home/tyfon/Downloads/index_branż.csv")
        .then(() => process.exit(0))
        .catch((error) => {
          console.error(error);
          process.exit(1);
        });
      process.exit(0);
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

export { seedCacheWskaznikow };
