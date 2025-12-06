/** biome-ignore-all lint/correctness/useUniqueElementIds: This is test */
import { Button } from "@bluevoid/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@bluevoid/ui/card";
import { Input } from "@bluevoid/ui/input";
import { Label } from "@bluevoid/ui/label";

function TestCard() {
	return (
		<Card className="flex gap-4 p-4">
			<Card className="w-96">
				<CardHeader>
					<CardTitle>Card Title</CardTitle>
					<CardDescription>Card Description</CardDescription>
				</CardHeader>
				<CardContent>
					<img src="/logo.svg" alt="test" className="bg-blue-200" />
				</CardContent>
				<CardFooter className="justify-end">
					<Button>Test</Button>
				</CardFooter>
			</Card>
			<Card className="w-96">
				<CardHeader>
					<CardTitle>Create project</CardTitle>
					<CardDescription>
						Deploy your new project in one-click.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<form>
						<div className="grid w-full items-center gap-4">
							<div className="flex flex-col space-y-1.5">
								<Label htmlFor="name">Name</Label>
								<Input id="name" placeholder="Name of your project" />
								<Input id="name" placeholder="Name of your project" />
								<Input id="name" placeholder="Name of your project" />
							</div>
						</div>
					</form>
				</CardContent>
				<CardFooter className="flex justify-between">
					<Button variant="outline">Cancel</Button>
					<Button>Deploy</Button>
				</CardFooter>
			</Card>
		</Card>
	);
}

export default TestCard;
