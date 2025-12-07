"use client";
import { useQuery } from "@tanstack/react-query";
import { orpc } from "@/lib/orpc";
import { useCacheDataTransform } from "../_hooks/useCacheDataTransform";
import { sections2 } from "../_utils/pkd";

function TestPage() {
  // const { data } = useQuery(
  //   orpc.cache_wskaznikow.queryOptions({ input: { pkds: sections2 } }),
  // );
  const { newData } = useCacheDataTransform(sections2, "firms");

  // const nowData = Object.groupBy(data ?? [], (val) => val.pkd_section ?? "");

  return <pre>{JSON.stringify(newData ?? {}).replaceAll("},", "},\n")}</pre>;
}
export default TestPage;
