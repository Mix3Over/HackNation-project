import { ScrollArea } from "@bluevoid/ui/scroll-area";
import { lorem } from "../lorem";
import { Separator } from "@bluevoid/ui/separator";
import { Fragment, useId } from "react";

const tags = Array.from({ length: 50 }).map(
	(_, i, a) => `v1.2.0-beta.${a.length - i}`,
);

function TestScrollArea() {
	const uuid = useId();
	return (
		<>
			<ScrollArea className="h-[200px] w-[350px] rounded-md border p-4">
				{lorem}
				{lorem}
			</ScrollArea>
			<ScrollArea className="h-72 w-48 rounded-md border">
				<div className="p-4">
					<h4 className="mb-4 font-medium text-sm leading-none">Tags</h4>
					{tags.map((tag) => (
						<Fragment key={`TestScrollArea${tag}:${uuid}`}>
							<div className="text-sm">{tag}</div>
							<Separator className="my-2" />
						</Fragment>
					))}
				</div>
			</ScrollArea>
		</>
	);
}

export default TestScrollArea;
