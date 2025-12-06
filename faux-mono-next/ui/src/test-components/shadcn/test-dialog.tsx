/** biome-ignore-all lint/correctness/useUniqueElementIds: This is test */

import { Button } from "@bluevoid/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@bluevoid/ui/dialog";
import { Input } from "@bluevoid/ui/input";
import { Label } from "@bluevoid/ui/label";

function TestDialog() {
	return (
		<Dialog>
			<DialogTrigger asChild>
				<Button variant="outline">Edit Profile</Button>
			</DialogTrigger>
			<DialogContent className="sm:max-w-[425px]">
				<DialogHeader>
					<DialogTitle>Edit profile</DialogTitle>
					<DialogDescription>
						{"Make changes to your profile here. Click save when you're done."}
					</DialogDescription>
				</DialogHeader>
				<div className="grid gap-4 py-4">
					<div className="grid grid-cols-4 items-center gap-4">
						<Label htmlFor="name" className="text-right">
							Name
						</Label>
						<Input
							id="name"
							defaultValue="Pedro Duarte"
							className="col-span-3"
						/>
					</div>
					<div className="grid grid-cols-4 items-center gap-4">
						<Label htmlFor="username" className="text-right">
							Username
						</Label>
						<Input
							id="username"
							defaultValue="@peduarte"
							className="col-span-3"
						/>
					</div>
				</div>
				<DialogFooter>
					<Button type="submit">Save changes</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

export default TestDialog;
