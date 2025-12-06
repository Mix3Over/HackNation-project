import { Button } from "@bluevoid/ui/button";
import { IconAlertCircle } from "@tabler/icons-react";

function TestButton() {
	return (
		<>
			<div className="flex gap-2">
				<Button>primary</Button>
				<Button variant="secondary">secondary</Button>
				<Button variant="outline">outline</Button>
				<Button variant="ghost">ghost</Button>
				<Button variant="link">link</Button>
			</div>
			<div className="flex gap-2">
				<Button size="sm">Size SM</Button>
				<Button size="default">Size default</Button>
				<Button size="lg">Size LG</Button>
				<Button size="icon">
					<IconAlertCircle />
				</Button>
			</div>
			<div className="flex gap-2">
				<Button disabled>primary</Button>
				<Button disabled variant="secondary">
					secondary
				</Button>
				<Button disabled variant="outline">
					outline
				</Button>
				<Button disabled variant="ghost">
					ghost
				</Button>
				<Button disabled variant="link">
					link
				</Button>
			</div>
			<div className="flex gap-2">
				<Button disabled size="sm">
					Size SM
				</Button>
				<Button disabled size="default">
					Size default
				</Button>
				<Button disabled size="lg">
					Size LG
				</Button>
				<Button disabled size="icon">
					<IconAlertCircle />
				</Button>
			</div>
		</>
	);
}

export default TestButton;
