import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { orpc } from "@/lib/orpc";

export type Level = "sektor" | "group";

export function useCacheDataTransform(
  pkds: string[],
  valueColumn: string,
  initialLevel: Level = "group",
) {
  const query = useQuery(
    orpc.cache_wskaznikow.queryOptions({ input: { pkds } }),
  );

  const [newData, legendX, legendY] = useMemo(() => {
    const groupByYear = Object.groupBy(query?.data ?? [], (val) => val.date);

    const newData: Record<number | string, any>[] = [];
    const legendX = [];
    let legendY = [];
    for (const group in groupByYear) {
      if (group.startsWith("S")) {
        continue;
      }
      legendX.push(group);
      const dataPoint: Record<any, any> = { year: group, date: group };
      for (const row of groupByYear[group as keyof typeof groupByYear] ?? []) {
        legendY.push(row.pkd ?? "");

        dataPoint[row.pkd ?? ""] = row[valueColumn as keyof typeof row];
      }
      newData.push(dataPoint);
    }
    legendY = [...new Set(legendY)];

    console.log(newData);

    return [
      newData,
      legendX.map((val) => ({ id: val, label: val, enabled: true })),
      legendY
        .sort((v1, v2) => Number.parseFloat(v1) - Number.parseFloat(v2))
        .map((val) => ({ id: val, label: val, enabled: true })),
    ];
  }, [query?.data, valueColumn]);

  return {
    ...query,
    newData,
    legendX,
    legendY,
  };
}
