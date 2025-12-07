"use client";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@bluevoid-test/ui/card";
import { Checkbox } from "@bluevoid-test/ui/checkbox";
import { Label } from "@bluevoid-test/ui/label";
import { ScrollArea } from "@bluevoid-test/ui/scroll-area";
import { useQuery } from "@tanstack/react-query";
import { useSelectedLayoutSegment } from "next/navigation";
import { Fragment, useEffect, useId, useMemo, useState } from "react";
import { orpc } from "@/lib/orpc";
import { Chart } from "./_components/Chart";
import ColumnSelector from "./_components/ColumnSelector";
import GroupFilterSelector from "./_components/GroupFilterSelector";
import LevelSelector from "./_components/LevelSelector";
import {
	type Level,
	useCacheDataTransform,
} from "./_hooks/useCacheDataTransform";
import { group_desc, groups } from "./_utils/group";
import { sections2 } from "./_utils/pkd";

const pkd_ids_groups = Object.keys(groups)
	.reduce((arr: string[], next: string) => {
		return [...arr, ...groups[next as keyof typeof groups]];
	}, [])
	.map((v) => ({ id: v, label: v, enabled: false }));

const first_in_group = Object.keys(groups).map(
	(key) => groups?.[key as keyof typeof groups]?.[0],
);

export type DataLegend = {
	id: string;
	label: string;
	enabled: boolean;
};

// TODO: preload some data
// TODO: precache all groupselections
// TODO: unshit this code

export function WskaznikiPage() {
	const uuid = useId();
	const [level, setLevel] = useState<Level>("group");
	const [selection, setSelection] = useState<DataLegend[]>(pkd_ids_groups);
	const [groupFilter, setGroupFilter] = useState<string>("A");
	const [column, setColumn] = useState("main_index");
	const selectionFilter = selection
		.filter((v) => v.enabled)
		.map((v) => v.label);
	console.log(selectionFilter);
	const { newData, legendX, legendY } = useCacheDataTransform(
		selectionFilter,
		column,
	);
	// biome-ignore lint/correctness/useExhaustiveDependencies: HACK
	useEffect(() => {
		if (level === "group") {
			onGroupFilterChange("A");
		}
	}, []);
	if (!newData) return null;

	const toggleItem = (label: string) => {
		setSelection((items) =>
			items.map((f) => (f.label === label ? { ...f, enabled: !f.enabled } : f)),
		);
	};

	const onGroupFilterChange = (groupFilter: string) => {
		setGroupFilter(groupFilter);
		setSelection((prev) =>
			prev.map((item) => ({
				...item,
				enabled: groups[groupFilter as keyof typeof groups].includes(
					item.label,
				),
			})),
		);
	};

	return (
		<div className="flex h-[calc(100vh-3.5rem)] flex-col gap-4 bg-blue-100 p-6 lg:flex-row">
			<div className="flex flex-1 flex-col gap-4">
				<Card>
					<CardContent className="grid grid-cols-3 gap-3">
						<LevelSelector onValueChange={setLevel} value={level} />
						<GroupFilterSelector
							disabled={level !== "group"}
							onValueChange={onGroupFilterChange}
							value={groupFilter}
						/>
						<ColumnSelector onValueChange={setColumn} value={column} />
					</CardContent>
				</Card>
				<Card className="grow">
					<CardContent>
						<Chart
							data={newData}
							dataLegend={selection}
							key={`${level}${groupFilter}${column}`}
						/>
					</CardContent>
				</Card>
			</div>

			<div className="w-full border-border border-l lg:w-80 xl:w-96">
				<Card>
					<CardHeader>
						<CardTitle className="text-base">Filtry PKD</CardTitle>
					</CardHeader>
					<CardContent>
						<ScrollArea className="h-[calc(100vh-3.5rem-9.5rem)]">
							<div className="space-y-4">
								{selection?.map((item, index) => (
									<Fragment key={`frag${item.id}`}>
										{first_in_group.includes(item.label) ? (
											<div className="space-y-4" key={`LETTER${item.id}`}>
												Sektor{" "}
												{
													Object.keys(groups)[
														first_in_group.indexOf(item.label)
													]
												}
											</div>
										) : null}
										<div className="flex items-center space-x-3" key={item.id}>
											<Checkbox
												checked={item.enabled}
												id={`${uuid}${item.id}`}
												onCheckedChange={() => toggleItem(item.label)}
											/>
											<Label
												className="flex cursor-pointer items-center gap-2 font-normal"
												htmlFor={`${uuid}${item.id}`}
											>
												<div
													className="h-3 w-3 shrink-0 rounded-full"
													style={{
														backgroundColor: `var(--chart-${(index % 10) + 1})`,
													}}
												/>
												{item.label}{" "}
												<span className="text-sm lowercase first-letter:capitalize">
													{group_desc?.[
														item.label as keyof typeof group_desc
													] ?? ""}
												</span>
											</Label>
										</div>
									</Fragment>
								))}
							</div>
						</ScrollArea>
					</CardContent>
				</Card>
			</div>
		</div>
	);
}

export default WskaznikiPage;
