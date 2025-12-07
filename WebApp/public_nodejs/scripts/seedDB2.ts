import { parse } from "csv-parse/sync";
import fs from "fs";
import { cache_wskaznikow } from "@/db/schema";
import { db } from "../db";

// CSV parsing and seeding function
async function seedCacheWskaznikow(
  csvFilePath: string = "/home/tyfon/Programming/HackNation-project/ProjectCore/Data/main_index_predictions.csv",
) {
  try {
    console.log("Reading CSV file...");
    const csvContent = fs.readFileSync(csvFilePath, "utf-8");

    console.log("Parsing CSV...");
    const records = parse(csvContent, { delimiter: ";" });

    console.log(`Parsed ${records.length} records`);

    const labelX = records[0];
    const newRecord = [];
    labelX[0] = "pkd";
    for (let x = 21; x < records[0].length; x++) {
      console.log(labelX[x]);
      for (let y = 1; y < records.length; y++) {
        const num = Number.parseFloat(records[y][x]);
        console.log(records[y][0], labelX[x]);
        const r: Record<any, any> = {
          date: labelX[x],
          id: crypto.randomUUID(),
          main_index: Number.isNaN(num) ? null : num,
          year: labelX[x],
          pkd: records[y][0],
          pkd_section: records[y][0],
          pkd_2025: records[y][0],
          pkd_name: "",
        };
        newRecord.push(r);
      }
    }
    console.log(newRecord);
    // // Generate unique ID as pkd_section + pkd_2025 + year
    // const preparedRecords: any = records.map((val: any) => ({
    //   ...val,
    //   year: val.year,
    //   date: val.year,
    //   pkd: val.pkd_section,
    //   id: crypto.randomUUID(),
    // }));

    // console.log(records);
    console.log("Clearing existing data...");
    // await db.delete(cache_wskaznikow);

    console.log("Inserting data...");
    for (const record of newRecord) {
      const data = await db
        .insert(cache_wskaznikow)
        .values(record as any)
        .execute();
      console.log(data);
    }

    console.log(
      `✅ Seeded ${newRecord.length} records into cache_wskaznikow table [file:1]`,
    );
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
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

export { seedCacheWskaznikow };
