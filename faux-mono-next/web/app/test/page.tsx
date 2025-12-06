"use client";
import { useQuery } from "@tanstack/react-query";
import { orpc } from "@/lib/orpc";

function TestPage() {
	const { data } = useQuery(orpc.testLoader.queryOptions());
	return <pre>{JSON.stringify(data ?? {}).replaceAll("},", "},\n")}</pre>;
}
export default TestPage;
